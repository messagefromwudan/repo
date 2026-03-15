"""
City Activity Advisor - AI-powered outing recommendations
Combines real weather data, local place discovery, and AI reasoning
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# Load environment variables first, before any other imports that might use them
from dotenv import load_dotenv
load_dotenv()

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Anthropic client after loading environment variables
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("ANTHROPIC_API_KEY not found in environment variables. Please set it in your .env file.")
client = anthropic.Anthropic(api_key=api_key)

# Initialize FastAPI app
app = FastAPI(
    title="City Activity Advisor",
    description="AI agent for personalized city activity recommendations",
    version="1.0.0"
)

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class WeatherRequest(BaseModel):
    """Request model for weather endpoint."""
    city: str
    days_ahead: int


class WeatherData(BaseModel):
    """Weather data model."""
    city: str
    temperature_max: float
    temperature_min: float
    weathercode: int
    wind_speed: float


class PlacesRequest(BaseModel):
    """Request model for places endpoint."""
    city: str
    activity: str


class PlaceInfo(BaseModel):
    """Individual place information."""
    name: str
    lat: float
    lon: float


class AdvisorRequest(BaseModel):
    """Request model for advisor endpoint."""
    city: str
    activity: str
    days_ahead: int


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint to verify the service is running.
    
    Returns:
        Dict with status and message confirming service availability.
    """
    return {"status": "ok", "data": {"message": "City Advisor is running"}}


async def geocode_city(city: str) -> Optional[Dict[str, float]]:
    """
    Convert city name to coordinates using Open-Meteo geocoding API.
    
    Args:
        city: Name of the city to geocode
        
    Returns:
        Dictionary with latitude and longitude, or None if not found
    """
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            return {"latitude": result["latitude"], "longitude": result["longitude"]}
        return None
    except Exception:
        return None


async def get_weather_forecast(latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
    """
    Get weather forecast from Open-Meteo API.
    
    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
        
    Returns:
        Weather forecast data, or None if failed
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,weathercode,windspeed_10m_max",
            "timezone": "auto"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


@app.post("/weather")
async def get_weather(request: WeatherRequest) -> Dict[str, Any]:
    """
    Get weather forecast for a city and specific day.
    
    Args:
        request: WeatherRequest with city and days_ahead
        
    Returns:
        Weather data or error message
    """
    try:
        # Validate days_ahead
        if request.days_ahead > 7:
            return {
                "status": "error",
                "data": {"message": "Weather forecast is only available up to 7 days ahead."}
            }
        
        if request.days_ahead < 0:
            return {
                "status": "error", 
                "data": {"message": "days_ahead must be 0 or greater."}
            }
        
        # Geocode city
        coords = await geocode_city(request.city)
        if not coords:
            return {
                "status": "error",
                "data": {"message": "City not found."}
            }
        
        # Get weather forecast
        forecast = await get_weather_forecast(coords["latitude"], coords["longitude"])
        if not forecast or "daily" not in forecast:
            return {
                "status": "error",
                "data": {"message": "Weather data unavailable."}
            }
        
        # Extract weather for the specific day
        daily_data = forecast["daily"]
        
        # Check if we have data for the requested day
        if request.days_ahead >= len(daily_data["time"]):
            return {
                "status": "error",
                "data": {"message": "Weather data not available for requested day."}
            }
        
        weather_data = {
            "city": request.city,
            "temperature_max": daily_data["temperature_2m_max"][request.days_ahead],
            "temperature_min": daily_data["temperature_2m_min"][request.days_ahead],
            "weathercode": daily_data["weathercode"][request.days_ahead],
            "wind_speed": daily_data["windspeed_10m_max"][request.days_ahead]
        }
        
        return {"status": "ok", "data": weather_data}
        
    except Exception as e:
        return {
            "status": "error",
            "data": {"message": f"Unexpected error: {str(e)}"}
        }


async def convert_activity_to_osm_tags(activity: str) -> Optional[Dict[str, str]]:
    """
    Convert activity description to OpenStreetMap tags using Claude AI.
    
    Args:
        activity: User's activity description (e.g., "billiards")
        
    Returns:
        Dictionary with key and value for OSM tags, or None if failed
    """
    try:
        logger.info(f"Converting activity '{activity}' to OSM tags")
        prompt = f"""You are an OpenStreetMap expert. Convert the activity '{activity}' 
to the correct OpenStreetMap tag pair.

Common examples:
- billiards/pool → {{"key": "leisure", "value": "billiard_hall"}}
- restaurant → {{"key": "amenity", "value": "restaurant"}}
- coffee shop → {{"key": "amenity", "value": "cafe"}}
- gym/fitness → {{"key": "leisure", "value": "fitness_centre"}}
- bar/pub → {{"key": "amenity", "value": "pub"}}
- cinema/movies → {{"key": "amenity", "value": "cinema"}}
- bowling → {{"key": "leisure", "value": "bowling_alley"}}
- park → {{"key": "leisure", "value": "park"}}
- hotel → {{"key": "tourism", "value": "hotel"}}
- museum → {{"key": "tourism", "value": "museum"}}

Respond with ONLY a valid raw JSON object, no markdown, no explanation.
Use the most commonly used OSM tag for this activity."""
        
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        raw_response = message.content[0].text
        
        # Clean the response by removing markdown code blocks
        cleaned_response = raw_response.strip()
        cleaned_response = cleaned_response.replace("```json", "").replace("```", "").strip()
        
        # Parse the JSON response
        import json
        tags = json.loads(cleaned_response)
        logger.info(f"Successfully converted activity to OSM tags: {tags}")
        return tags
    except Exception as e:
        logger.error(f"Error converting activity to OSM tags: {str(e)}")
        return None


async def get_city_bounding_box(city: str) -> Optional[Dict[str, float]]:
    """
    Get bounding box for a city using Nominatim API.
    
    Args:
        city: Name of the city
        
    Returns:
        Dictionary with south, west, north, east bounds, or None if not found
    """
    try:
        logger.info(f"Getting bounding box for city: {city}")
        url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
        headers = {"User-Agent": "City-Activity-Advisor/1.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if len(data) > 0:
            result = data[0]
            bbox = result["boundingbox"]
            # bbox format: [south, north, west, east]
            bounding_box = {
                "south": float(bbox[0]),
                "west": float(bbox[2]), 
                "north": float(bbox[1]),
                "east": float(bbox[3])
            }
            logger.info(f"Successfully got bounding box for {city}")
            return bounding_box
        return None
    except Exception as e:
        logger.error(f"Error getting bounding box for {city}: {str(e)}")
        return None


async def get_places_from_claude(city: str, activity: str) -> Optional[list]:
    """
    Get place suggestions from Claude when OpenStreetMap has no data.
    
    Args:
        city: Name of the city
        activity: Activity to find places for
        
    Returns:
        List of AI-suggested places, or None if failed
    """
    try:
        logger.info(f"Getting places from Claude for {activity} in {city}")
        prompt = f"""List 3-5 real places in {city} where someone can do {activity}. 
Respond ONLY with raw JSON, no markdown, no explanation, in this exact format:
[
  {{"name": "Place Name", "address": "Street address or area", "note": "One short tip"}},
]"""
        
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=200,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        raw_response = message.content[0].text
        
        # Clean the response by removing markdown code blocks
        cleaned_response = raw_response.strip()
        cleaned_response = cleaned_response.replace("```json", "").replace("```", "").strip()
        
        # Parse the JSON response
        import json
        places = json.loads(cleaned_response)
        
        # Add source field to each place
        for place in places:
            place["source"] = "ai_suggested"
        
        logger.info(f"Successfully got {len(places)} places from Claude")
        return places
    except Exception as e:
        logger.error(f"Error getting places from Claude: {str(e)}")
        return None


async def find_places_with_osm_tags(key: str, value: str, bbox: Dict[str, float]) -> Optional[list]:
    """
    Find places using OpenStreetMap Overpass API.
    
    Args:
        key: OSM tag key (e.g., "leisure")
        value: OSM tag value (e.g., "billiards")
        bbox: Bounding box with south, west, north, east
        
    Returns:
        List of places with name and coordinates, or None if failed
    """
    try:
        logger.info(f"Searching OSM for {key}={value}")
        
        query = f"""
[out:json];
(
  node["{key}"="{value}"]({bbox["south"]},{bbox["west"]},{bbox["north"]},{bbox["east"]});
  way["{key}"="{value}"]({bbox["south"]},{bbox["west"]},{bbox["north"]},{bbox["east"]});
  relation["{key}"="{value}"]({bbox["south"]},{bbox["west"]},{bbox["north"]},{bbox["east"]});
);
out center 10;
"""
        
        url = "https://overpass-api.de/api/interpreter"
        headers = {"Content-Type": "text/plain"}
        response = requests.post(url, data=query, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        elements = data.get("elements", [])
        
        places = []
        
        for element in elements:
            # Handle coordinates for node vs way/relation
            if element["type"] == "node":
                lat = element["lat"]
                lon = element["lon"]
            elif "center" in element:
                lat = element["center"]["lat"]
                lon = element["center"]["lon"]
            else:
                continue
            
            # Handle name with fallback
            name = element.get("tags", {}).get("name", "Unnamed Place")
            
            place = {
                "name": name,
                "lat": lat,
                "lon": lon,
                "source": "openstreetmap"
            }
            places.append(place)
        
        logger.info(f"Found {len(places)} OSM places")
        return places
    except Exception as e:
        logger.error(f"Error searching OSM places: {str(e)}")
        return None


@app.post("/places")
async def get_places(request: PlacesRequest) -> Dict[str, Any]:
    """
    Find places for a specific activity in a city.
    
    Args:
        request: PlacesRequest with city and activity
        
    Returns:
        List of places or error message
    """
    try:
        logger.info(f"Processing places request for {request.activity} in {request.city}")
        
        # Step 1: Convert activity to OSM tags using Claude
        tags = await convert_activity_to_osm_tags(request.activity)
            
        if not tags or "key" not in tags or "value" not in tags:
            return {
                "status": "error",
                "data": {"message": f"Failed to convert activity '{request.activity}' to OSM tags."}
            }
        
        # Step 2: Get city bounding box
        bbox = await get_city_bounding_box(request.city)
            
        if not bbox:
            return {
                "status": "error",
                "data": {"message": "City not found."}
            }
        
        # Step 3: Find places using Overpass API
        places = await find_places_with_osm_tags(tags["key"], tags["value"], bbox)
            
        if places is None:
            return {
                "status": "error",
                "data": {"message": "Places service is currently unavailable."}
            }
        
        # Step 4: Fallback to Claude if no OSM places found
        if len(places) == 0:
            logger.info(f"No OSM places found, trying Claude fallback")
            claude_places = await get_places_from_claude(request.city, request.activity)
                
            if claude_places and len(claude_places) > 0:
                places = claude_places
            else:
                return {
                    "status": "error",
                    "data": {"message": "No places found for that activity in this city."}
                }
        
        response_data = {
            "city": request.city,
            "activity": request.activity,
            "places": places
        }
        
        logger.info(f"Successfully processed places request for {request.city}")
        return {"status": "ok", "data": response_data}
        
    except Exception as e:
        logger.error(f"Error in places endpoint: {str(e)}")
        return {
            "status": "error",
            "data": {"message": f"Unexpected error: {str(e)}"}
        }


async def generate_recommendation(city: str, activity: str, days_ahead: int, weather: Dict[str, Any], places: list) -> str:
    """
    Generate personalized recommendation using Claude AI.
    
    Args:
        city: Name of the city
        activity: Activity to recommend
        days_ahead: Days ahead for the activity
        weather: Weather data dictionary
        places: List of places
        
    Returns:
        Personalized recommendation text
    """
    try:
        logger.info(f"Generating recommendation for {activity} in {city}")
        
        # Format places list nicely
        places_text = ""
        for i, place in enumerate(places, 1):
            if place.get("source") == "openstreetmap":
                places_text += f"{i}. {place['name']} (OSM data)\n"
            else:
                places_text += f"{i}. {place['name']} - {place.get('address', 'Address unknown')} - {place.get('note', '')}\n"
        
        prompt = f"""The user wants to do {activity} in {city} in {days_ahead} day(s).
   
Weather forecast:
- Max temperature: {weather['temperature_max']}°C
- Min temperature: {weather['temperature_min']}°C
- Wind speed: {weather['wind_speed']} km/h
- Weather code: {weather['weathercode']}
   
Places they could go:
{places_text}
   
Give a friendly, personalized recommendation that includes:
1. Which place to go and why
2. What to wear based on the weather
3. Best time of day to go
4. One practical tip
   
Be specific, warm, and helpful. Keep it under 150 words."""
        
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=250,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        recommendation = message.content[0].text.strip()
        logger.info(f"Successfully generated recommendation")
        return recommendation
    except Exception as e:
        logger.error(f"Error generating recommendation: {str(e)}")
        return "Recommendation unavailable at this time."


@app.post("/advisor")
async def get_advisor(request: AdvisorRequest) -> Dict[str, Any]:
    """
    Get comprehensive activity recommendation combining weather, places, and AI advice.
    
    Args:
        request: AdvisorRequest with city, activity, and days_ahead
        
    Returns:
        Complete recommendation with weather, places, and AI advice
    """
    try:
        logger.info(f"Processing advisor request for {request.activity} in {request.city} in {request.days_ahead} days")
        
        # Step 1: Get weather data
        weather_request = WeatherRequest(city=request.city, days_ahead=request.days_ahead)
        weather_response = await get_weather(weather_request)
            
        if weather_response["status"] != "ok":
            return weather_response
        
        weather_data = weather_response["data"]
        
        # Step 2: Get places data
        places_request = PlacesRequest(city=request.city, activity=request.activity)
        places_response = await get_places(places_request)
            
        if places_response["status"] != "ok":
            return places_response
        
        places_data = places_response["data"]["places"]
        
        # Step 3: Generate recommendation
        recommendation = await generate_recommendation(
            request.city, 
            request.activity, 
            request.days_ahead, 
            weather_data, 
            places_data
        )
        
        # Combine all results
        response_data = {
            "city": request.city,
            "activity": request.activity,
            "weather": weather_data,
            "places": places_data,
            "recommendation": recommendation
        }
        
        logger.info(f"Successfully processed advisor request for {request.city}")
        return {"status": "ok", "data": response_data}
        
    except Exception as e:
        logger.error(f"Error in advisor endpoint: {str(e)}")
        return {
            "status": "error",
            "data": {"message": f"Unexpected error: {str(e)}"}
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

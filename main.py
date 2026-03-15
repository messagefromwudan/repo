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

# Load environment variables immediately after dotenv
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY")
GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check API keys with warnings instead of errors
if not ANTHROPIC_API_KEY:
    logging.warning("ANTHROPIC_API_KEY not found in environment variables. Please set it in your .env file.")

if not FOURSQUARE_API_KEY:
    logging.warning("FOURSQUARE_API_KEY not found in environment variables. Please set it in your .env file.")

if not GEOAPIFY_API_KEY:
    logging.warning("GEOAPIFY_API_KEY not found in environment variables. Please set it in your .env file.")

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import anthropic

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
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


class DetailsRequest(BaseModel):
    """Request model for details endpoint."""
    city: str
    place_name: str


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


GEOAPIFY_CATEGORIES = {
    "cafe": "catering.cafe",
    "coffee": "catering.cafe",
    "matcha": "catering.cafe",
    "tea": "catering.cafe",
    "restaurant": "catering.restaurant",
    "food": "catering.restaurant",
    "bar": "catering.bar",
    "pub": "catering.bar",
    "fast food": "catering.fast_food",
    "fitness": "sport",
    "gym": "sport",
    "karting": "sport",
    "go kart": "sport",
    "sport": "sport",
    "billiards": "entertainment",
    "pool": "entertainment",
    "bowling": "entertainment.bowling_alley",
    "arcade": "entertainment.amusement_arcade",
    "cinema": "entertainment.cinema",
    "movie": "entertainment.cinema",
    "park": "leisure",
    "museum": "tourism.museum",
    "hotel": "accommodation.hotel",
}


def get_geoapify_category(activity: str) -> str:
    """
    Convert activity to Geoapify category using keyword matching.
    
    Args:
        activity: User's activity description
        
    Returns:
        Geoapify category string
    """
    try:
        logger.info(f"Converting activity '{activity}' to Geoapify category")
        
        # Lowercase the activity for matching
        activity_lower = activity.lower()
        
        # Check for keyword matches
        for keyword, category in GEOAPIFY_CATEGORIES.items():
            if keyword in activity_lower:
                logger.info(f"Matched keyword '{keyword}' to category '{category}'")
                return category
        
        # Default to restaurant if no match found
        logger.info(f"No keyword match found, defaulting to 'catering.restaurant'")
        return "catering.restaurant"
        
    except Exception as e:
        logger.error(f"Error converting activity to Geoapify category: {str(e)}")
        return "catering.restaurant"


async def get_places_from_claude(city: str, activity: str, city_lat: float, city_lon: float) -> Optional[list]:
    """
    Get place suggestions from Claude when Geoapify results don't match.
    
    Args:
        city: Name of the city
        activity: Activity to find places for
        city_lat: City center latitude
        city_lon: City center longitude
        
    Returns:
        List of AI-suggested places, or None if failed
    """
    try:
        logger.info(f"Getting places from Claude for {activity} in {city}")
        prompt = f"""List 3-5 real places in {city} where someone can do {activity}.
Respond ONLY with raw JSON, no markdown:
[{{"name": "Place Name", "address": "Street address", "category": "Type of place"}}]"""
        
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
        
        # Add source and coordinates to each place
        for place in places:
            place["source"] = "ai_suggested"
            place["lat"] = city_lat
            place["lon"] = city_lon
            place["google_maps_url"] = f"https://www.google.com/maps/search/{place['name'].replace(' ', '+')}+{city.replace(' ', '+')}"
            place["rating"] = None
        
        logger.info(f"Successfully got {len(places)} places from Claude")
        return places
    except Exception as e:
        logger.error(f"Error getting places from Claude: {str(e)}")
        return None


NICHE_ACTIVITIES = [
    "karting", "go kart", "billiards", "pool", "snooker",
    "escape room", "laser tag", "paintball", "trampoline",
    "climbing", "archery", "mini golf", "axe throwing",
    "virtual reality", "vr", "bowling"
]


async def find_places_with_geoapify(city: str, activity: str) -> Optional[list]:
    """
    Find places using Geoapify Places API or Claude fallback for niche activities.
    
    Args:
        city: Name of the city
        activity: Activity to search for
        
    Returns:
        List of places with detailed information, or None if failed
    """
    try:
        logger.info(f"Searching Geoapify for {activity} in {city}")
        logger.info(f"Geoapify key loaded: {bool(GEOAPIFY_API_KEY)}")
        
        # Check if this is a niche activity that should bypass Geoapify
        activity_lower = activity.lower()
        if any(niche in activity_lower for niche in NICHE_ACTIVITIES):
            logger.info(f"Niche activity detected: {activity}, skipping Geoapify, using Claude fallback")
            # We need city coordinates first for Claude fallback
            geocode_url = "https://api.geoapify.com/v1/geocode/search"
            geocode_params = {
                "text": city,
                "apiKey": GEOAPIFY_API_KEY,
                "limit": 1
            }
            
            geocode_response = requests.get(geocode_url, params=geocode_params, timeout=10)
            geocode_response.raise_for_status()
            
            geocode_data = geocode_response.json()
            features = geocode_data.get("features", [])
            
            if not features:
                logger.error(f"City '{city}' not found")
                return None
            
            coordinates = features[0]["geometry"]["coordinates"]
            city_lon, city_lat = coordinates
            
            return await get_places_from_claude(city, activity, city_lat, city_lon)
        
        # Step 1: Geocode the city
        geocode_url = "https://api.geoapify.com/v1/geocode/search"
        geocode_params = {
            "text": city,
            "apiKey": GEOAPIFY_API_KEY,
            "limit": 1
        }
        
        logger.info(f"Geoapify geocode params: {geocode_params}")
        
        geocode_response = requests.get(geocode_url, params=geocode_params, timeout=10)
        geocode_response.raise_for_status()
        
        geocode_data = geocode_response.json()
        features = geocode_data.get("features", [])
        
        if not features:
            logger.error(f"City '{city}' not found")
            return None
        
        # Get coordinates (note: Geoapify returns [lon, lat])
        coordinates = features[0]["geometry"]["coordinates"]
        city_lon, city_lat = coordinates
        
        logger.info(f"Geocoded {city} to: {city_lat}, {city_lon}")
        
        # Step 2: Convert activity to Geoapify category
        category = get_geoapify_category(activity)
        if not category:
            logger.error(f"Failed to convert activity to Geoapify category")
            return None
        
        # Step 3: Search for places
        places_url = "https://api.geoapify.com/v2/places"
        places_params = {
            "categories": category,
            "filter": f"circle:{city_lon},{city_lat},3000",
            "limit": 5,
            "apiKey": GEOAPIFY_API_KEY
        }
        
        # Debug logging
        logger.info(f"Geoapify places URL: {places_url}")
        logger.info(f"Geoapify places params: {places_params}")
        
        places_response = requests.get(places_url, params=places_params, timeout=15)
        places_response.raise_for_status()
        
        places_data = places_response.json()
        place_features = places_data.get("features", [])
        
        places = []
        
        for feature in place_features:
            # Extract place information
            properties = feature.get("properties", {})
            name = properties.get("name", "")
            
            # Filter out unnamed places
            if not name or name == "Unnamed Place":
                continue
            
            # Get coordinates (note: Geoapify returns [lon, lat])
            place_coordinates = feature.get("geometry", {}).get("coordinates", [0, 0])
            place_lon, place_lat = place_coordinates
            
            place = {
                "name": name,
                "address": properties.get("formatted", "Address not available"),
                "rating": None,  # Geoapify doesn't provide ratings in free tier
                "category": properties.get("categories", [""])[0] or "Place",
                "lat": place_lat,
                "lon": place_lon,
                "google_maps_url": f"https://www.google.com/maps/search/?api=1&query={place_lat},{place_lon}",
                "source": "geoapify"
            }
            places.append(place)
        
        logger.info(f"Found {len(places)} Geoapify places")
        
        # Step 4: Check if results make sense with Claude
        if places:
            place_names = [f"{p['name']} ({p.get('category', 'Place')})" for p in places]
            places_list = "\n".join(place_names)
            
            relevance_prompt = f"""Do any of these places relate to '{activity}'?
Places: {places_list}
Respond with only 'yes' or 'no'."""
            
            try:
                relevance_message = client.messages.create(
                    model="claude-haiku-4-5",
                    max_tokens=10,
                    messages=[
                        {"role": "user", "content": relevance_prompt}
                    ]
                )
                
                relevance_response = relevance_message.content[0].text.strip().lower()
                logger.info(f"Claude relevance check: {relevance_response}")
                
                if relevance_response == 'no':
                    logger.info(f"Claude says places don't match, falling back to Claude suggestions")
                    return await get_places_from_claude(city, activity, city_lat, city_lon)
                    
            except Exception as e:
                logger.error(f"Error in relevance check: {str(e)}")
                # Continue with Geoapify results if check fails
        
        return places[:5]  # Return max 5 results
        
    except Exception as e:
        logger.error(f"Error searching Geoapify places: {str(e)}")
        return None


async def filter_places_by_offering(places: list, activity: str, city: str) -> list:
    """
    Filter places using Claude AI to determine which ones offer the requested activity.
    
    Args:
        places: List of places from Geoapify
        activity: Activity the user is looking for
        city: City name for context
        
    Returns:
        Filtered list of places, or original list if filtering fails
    """
    try:
        logger.info(f"Filtering places for {activity} in {city}")
        
        # Create a list of place names and categories
        places_info = []
        for place in places:
            places_info.append(f"{place['name']} ({place.get('category', 'Place')})")
        
        places_list = "\n".join(places_info)
        
        prompt = f"""The user is looking for '{activity}' in {city}.
Here are some places: {places_list}

Remove ONLY places that are completely unrelated (e.g. user wants karting, 
remove swimming pools and libraries).
Keep anything that could even loosely relate to the activity.
If fewer than 2 places remain after filtering, return ALL original places instead.

Respond with raw JSON array of place names to keep."""
        
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
        places_to_keep = json.loads(cleaned_response)
        
        # Filter the original places list
        filtered_places = []
        for place in places:
            if place["name"] in places_to_keep:
                filtered_places.append(place)
        
        # If fewer than 2 places remain, return all original places
        if len(filtered_places) < 2:
            logger.info(f"Only {len(filtered_places)} places remain after filtering, returning all {len(places)} original places")
            return places
        
        logger.info(f"Filtered {len(places)} places to {len(filtered_places)} relevant ones")
        return filtered_places
        
    except Exception as e:
        logger.error(f"Error filtering places: {str(e)}")
        # If filtering fails, return the original list
        return places


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
        
        # Step 1: Find places using Geoapify API
        places = await find_places_with_geoapify(request.city, request.activity)
            
        if not places:
            return {
                "status": "error",
                "data": {"message": "No places found for that activity in this city."}
            }
        
        # Step 2: Filter places using Claude to ensure they offer the requested activity
        filtered_places = await filter_places_by_offering(places, request.activity, request.city)
        
        response_data = {
            "city": request.city,
            "activity": request.activity,
            "places": filtered_places
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


def clean_activity(activity: str) -> str:
    """
    Clean activity string by removing time words and filler words.
    
    Args:
        activity: Raw activity string from user
        
    Returns:
        Cleaned activity string
    """
    import re
    
    # Convert to lowercase for processing
    cleaned = activity.lower().strip()
    
    # Remove time-related phrases
    time_patterns = [
        r'\btomorrow\b',
        r'\btoday\b', 
        r'\btonight\b',
        r'\bthis evening\b',
        r'\bin \d+ days?\b',
        r'\bnext week\b'
    ]
    
    for pattern in time_patterns:
        cleaned = re.sub(pattern, '', cleaned)
    
    # Remove filler phrases
    filler_patterns = [
        r'\bi want to\b',
        r'\bfind me\b',
        r'\blooking for\b',
        r'\bi want to go\b'
    ]
    
    for pattern in filler_patterns:
        cleaned = re.sub(pattern, '', cleaned)
    
    # Clean up extra spaces and return
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned if cleaned else activity


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
        
        # Clean the activity for places API
        cleaned_activity = clean_activity(request.activity)
        logger.info(f"Cleaned activity: '{cleaned_activity}'")
        
        # Step 1: Get weather data
        weather_request = WeatherRequest(city=request.city, days_ahead=request.days_ahead)
        weather_response = await get_weather(weather_request)
            
        if weather_response["status"] != "ok":
            return weather_response
        
        weather_data = weather_response["data"]
        
        # Step 2: Get places data with cleaned activity
        places_request = PlacesRequest(city=request.city, activity=cleaned_activity)
        places_response = await get_places(places_request)
            
        if places_response["status"] != "ok":
            return places_response
        
        places_data = places_response["data"]["places"]
        
        # Step 3: Generate recommendation
        recommendation = await generate_recommendation(
            request.city, 
            request.activity,  # Keep original activity for recommendation text
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


async def get_place_details(city: str, place_name: str) -> Optional[str]:
    """
    Get detailed information about a specific place using Claude AI.
    
    Args:
        city: Name of the city
        place_name: Name of the specific place
        
    Returns:
        Detailed place information as string, or None if failed
    """
    try:
        logger.info(f"Getting details for {place_name} in {city}")
        prompt = f"""Give practical information about {place_name} in {city}. Include:
   1. What kind of place it is
   2. What to expect when you visit
   3. Price range (if known)
   4. Best time to visit
   5. One insider tip
   Be friendly and specific. Keep it under 120 words."""
        
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=200,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        details = message.content[0].text.strip()
        logger.info(f"Successfully got details for {place_name}")
        return details
    except Exception as e:
        logger.error(f"Error getting place details: {str(e)}")
        return None


@app.post("/details")
async def get_place_details_endpoint(request: DetailsRequest) -> Dict[str, Any]:
    """
    Get detailed information about a specific place in a city.
    
    Args:
        request: DetailsRequest with city and place_name
        
    Returns:
        Place details or error message
    """
    try:
        logger.info(f"Processing details request for {request.place_name} in {request.city}")
        
        # Get place details from Claude
        details = await get_place_details(request.city, request.place_name)
            
        if not details:
            return {
                "status": "error",
                "data": {"message": "Place details unavailable at this time."}
            }
        
        response_data = {
            "place_name": request.place_name,
            "city": request.city,
            "details": details
        }
        
        logger.info(f"Successfully processed details request for {request.place_name}")
        return {"status": "ok", "data": response_data}
        
    except Exception as e:
        logger.error(f"Error in details endpoint: {str(e)}")
        return {
            "status": "error",
            "data": {"message": f"Unexpected error: {str(e)}"}
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
tools.py

Deterministic, hardcoded tool functions the agent can call. No real API
calls  mock-first, same principle as the rest of this repo's evaluators.
"""

WEATHER_DATA = {
    "Mumbai": "Humid and cloudy",
    "Delhi": "Clear skies",
    "Bangalore": "Pleasant and breezy",
    "Chennai": "Hot and humid",
    "Pune": "Mild with light showers",
}

TEMPERATURE_DATA = {
    "Mumbai": 32.5,
    "Delhi": 38.0,
    "Bangalore": 24.0,
    "Chennai": 34.5,
    "Pune": 27.0,
}

POPULATION_DATA = {
    "Mumbai": 20_411_000,
    "Delhi": 32_941_000,
    "Bangalore": 13_193_000,
    "Chennai": 11_324_000,
    "Pune": 7_400_000,
}


def get_weather(city: str) -> str:
    return WEATHER_DATA.get(city, f"Weather data unavailable for {city}")


def get_temperature(city: str) -> float:
    return TEMPERATURE_DATA.get(city, -1.0)


def get_population(city: str) -> int:
    return POPULATION_DATA.get(city, 0)

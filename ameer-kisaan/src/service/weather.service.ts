import { Injectable } from '@nestjs/common';
import { WeatherAccessor } from '../accessor/weather.accessor';
import { WeatherDocument } from '../schema/weather.schema';
import { WeatherResponseDto } from '../dto/weather.dto';

@Injectable()
export class WeatherService {
	constructor(private readonly weatherAccessor: WeatherAccessor) {}

	async getAllWeatherData(): Promise<WeatherResponseDto[]> {
		try {
			const weatherData = await this.weatherAccessor.findAll();
			console.log('Service Data:', weatherData);
			return weatherData.map(data => this.toResponseDto(data));
		} catch (error) {
			console.error('Error getting weather data:', error);
			throw error;
		}
	}
	
	async getWeatherByLocation(lat: Number, lon: Number): Promise<WeatherResponseDto | undefined | null> {
		try {
			const weatherData = await this.weatherAccessor.findByLocation(lat, lon);
			return weatherData ? this.toResponseDto(weatherData) : null;
		} catch (error) {
			console.error(`Error getting weather data by location (${lat}, ${lon}):`, error);
			throw error;
		}
	}

	private toResponseDto(weatherDocument: WeatherDocument): WeatherResponseDto {
		return {
			lat: weatherDocument.lat,
			lon: weatherDocument.lon,
			temperature: weatherDocument.temperature,
			humidity: weatherDocument.humidity,
			description: weatherDocument.description,
			timestamp: weatherDocument.timestamp
		}
	}
}
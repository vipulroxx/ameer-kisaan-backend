import { Controller, Get, Param, Query } from '@nestjs/common';
import { WeatherService } from '../service/weather.service';
import { WeatherResponseDto, WeatherRequestDto } from '../dto/weather.dto';

@Controller('weather')
export class WeatherController {
	constructor(private readonly weatherService: WeatherService) {}

	@Get('all')
	async getAllWeatherData(): Promise<WeatherResponseDto[]> {
		try {
			return await this.weatherService.getAllWeatherData();
		} catch (error) {
			console.error('Error fetching all weather data in controller', error);
			throw (error);
		}
	}

	@Get('by-location')
	async getWeatherByLocation(@Query() query: WeatherRequestDto): Promise<WeatherResponseDto | undefined | null> {
		try {
			return this.weatherService.getWeatherByLocation(query.lat, query.lon);
		} catch (error) {
			console.error(`Error fetching weather data by location (${query.lat}, ${query.lon})`, error);
		}
	}
}
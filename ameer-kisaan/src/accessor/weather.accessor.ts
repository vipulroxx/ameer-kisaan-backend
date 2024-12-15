import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { WeatherDocument } from '../schema/weather.schema';

@Injectable()
export class WeatherAccessor {
	constructor(@InjectModel('Weather') private readonly weatherModel: Model<WeatherDocument>) {}

	async findAll(): Promise<WeatherDocument[]> {
		try {
			const data = await this.weatherModel.find().exec();
			console.log('Fetched Data:', data);
			return data;
		} catch (error) {
			console.error('Error fetching all weather data:', error);
			throw error;
		}
	}
	
	async findByLocation(lat: Number, lon: Number): Promise<WeatherDocument | undefined | null> {
		try {
			return await this.weatherModel.findOne({ lat, lon }).exec();
		} catch (error) {
			console.error('Error fetching weather data for location (${lat}, ${lon}', error);
			throw error;
		}
	}
}
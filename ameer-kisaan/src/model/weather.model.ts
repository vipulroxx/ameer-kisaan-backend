import { Model } from 'mongoose';
import { WeatherDocument } from '../schema/weather.schema';

export class WeatherModel {
	constructor(private readonly weatherModel: Model<WeatherDocument>) {}

	async save(weatherData: WeatherDocument): Promise<WeatherDocument> {
		try {
			const newDocument = new this.weatherModel(weatherData);
			return await newDocument.save();
		} catch (error: unknown) {
			console.error("Error saving the document");
			throw error;
		}
	}

	async findAll(): Promise<WeatherDocument[]> {
		try {
			return await this.weatherModel.find().exec();
		} catch (error: unknown) {
			console.error('Error get all documents');
			throw error;
		}
	}
}
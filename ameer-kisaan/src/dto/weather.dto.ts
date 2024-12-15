export class WeatherRequestDto {
	lat: Number;
	lon: Number;

	constructor(lat: Number = 0.0, lon: Number = 0.0) {
		this.lat = lat;
		this.lon = lon;
	}
}

export interface WeatherResponseDto {
	lat: Number;
	lon: Number;
	temperature: Number;
	humidity: Number;
	description: String;
	timestamp: Date;
}
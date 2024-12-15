import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { WeatherController } from '../controller/weather.controller';
import { WeatherService } from '../service/weather.service';
import { WeatherAccessor } from '../accessor/weather.accessor';
import { WeatherSchema } from '../schema/weather.schema';

@Module({
	imports: [
		MongooseModule.forFeature([{ name: "Weather", schema: WeatherSchema }]),
	],
	controllers: [WeatherController],
	providers: [WeatherService, WeatherAccessor],
})

export class WeatherModule {}
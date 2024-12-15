import { Module } from "@nestjs/common";
import { MongooseModule} from "@nestjs/mongoose";
import { WeatherModule } from './src/module/weather.module';

@Module({
  imports: [
    MongooseModule.forRootAsync({
      useFactory: () => {
        console.log('Connecting to MongoDB...');
        return {
          uri: 'mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.3.2',
          dbName: 'ameer-kisaan',
          retryAttempts: 5, 
          retryDelay: 3000,
          verboseRetryLog: true,
        }
      },
    }),
    WeatherModule,
  ],
})
export class AppModule {}

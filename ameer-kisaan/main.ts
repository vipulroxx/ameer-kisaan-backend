import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
	try {
		const app = await NestFactory.create(AppModule);
		await app.listen(5001);
		console.log('Started at port 5001');
	} catch (error) {
		console.error('Error starting the NestJS application', error);
	}
}

bootstrap();
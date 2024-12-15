import { Document } from 'mongoose';
import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';

@Schema()
export class Weather {
  @Prop({ required: true })
  lat!: Number;

  @Prop({ required: true })
	lon!: Number;

  @Prop({ required: true })
	temperature!: Number;

  @Prop({ required: true })
	humidity!: Number;

  @Prop({ required: true })
	description!: string;

  @Prop({ required: true })
	timestamp!: Date;
}

export const WeatherSchema = SchemaFactory.createForClass(Weather);

export type WeatherDocument = Weather & Document;

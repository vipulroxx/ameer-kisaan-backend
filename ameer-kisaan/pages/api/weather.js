// const fetchWeatherData = async (lattitude, longitude) => {
// 	try {
// 		const [currentResponse, historicalResponse, forecastResponse] = await Promise.all([
// 			fetch(`https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,wind_speed_10m`),
//       fetch(`https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&past_days=10&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m`),
//       fetch(`https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&daily=temperature_2m_max,temperature_2m_min`)
//     ]);

// 		if (!currentResponse.ok || !historicalResponse.ok || !forecastResponse) {
// 			throw new Error('Failed to fetch weather data');
// 		}

// 		const currentWeather = await currentResponse.json();
// 		const historicalWeather = await historicalResponse.json();
// 		const forecastWeather = await forecastResponse.json();

// 		return {
// 			currentWeather: currentWeather.current,
// 			historicalWeather: historicalWeather.hourly,
// 			forecastWeather: forecastWeather.daily,
// 		};
// 	} catch (error) {
// 			throw new Error(`Error fetching weather data: ${error.message}`);
// 	}
// };

// export default async function handler(req, res) {
// 	const { latitude, longitude } = req.query;

// 	if (!latitude || !longitude) {
// 		return res.status(400).json({ error: 'Latitude and Longitude are required'});
// 	}

// 	try {
// 		const weatherData = await fetchWeatherData(latitude, longitude);
// 		res.status(200).json(weatherData);
// 	} catch(error) {
// 		res.status(500).json({ error: error.message });
// 	}
// }
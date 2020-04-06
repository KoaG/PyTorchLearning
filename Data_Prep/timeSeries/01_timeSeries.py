import csv
import numpy as np
import torch

data_path ="./hour_fixed.csv"
bikes_np = np.loadtxt(data_path, dtype=np.float32, delimiter=',', skiprows=1, converters={1: lambda x: float(x[8:10])})
#? converters -> converts date string to number corresponding to day of the month
bikes = torch.from_numpy(bikes_np)
#print(bikes)
print(bikes.shape, bikes.stride())

#* Changing 2D tensor to 3D
daily_bikes = bikes.view(-1, 24, bikes.shape[1])
print(daily_bikes.shape, daily_bikes.stride())
#* To change the tensor to NxCxL, we need to tanspose along L and C
daily_bikes = daily_bikes.transpose(1, 2)
print(daily_bikes.shape, daily_bikes.stride())

#* One-hot encoding for only day 1
first_day = bikes[:24].long()
print(first_day.shape)
weather_onehot = torch.zeros(first_day.shape[0], 4)
print(first_day[:,9])
weather_onehot.scatter_(
	dim=1,
	index=first_day[:,9].unsqueeze(1) - 1, #? -1 because indices are 0 based 1-4 -> 0-3
	value=1.0)
print(weather_onehot)
print(torch.cat((bikes[:24], weather_onehot), 1).shape)
print(torch.cat((bikes[:24], weather_onehot), 1)[:1])

#* One-hot all weather data for bikes
weather_onehot = torch.zeros(bikes.shape[0], 4)
weather_onehot.scatter_(
	dim=1,
	index=bikes[:,9].long().unsqueeze(1) - 1,
	value=1.0)
print(weather_onehot.shape)
print(torch.cat((bikes, weather_onehot), 1).shape)
print(torch.cat((bikes, weather_onehot), 1)[:1])

#* Doing one-hot with daily bikes data
daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4, daily_bikes.shape[2]) #* 730 x 4 x 24
print(daily_weather_onehot.shape)

daily_weather_onehot.scatter_(
	dim=1,
	index=daily_bikes[:,9,:].long().unsqueeze(1) - 1,
	value=1.0)
print(daily_weather_onehot.shape)

daily_bikes = torch.cat((daily_bikes, daily_weather_onehot), dim=1)
print(daily_bikes.shape)
print(daily_bikes[0,:,0])

#* If labels have ordinal relationship, we can map the variable -> [0.0 - 1.0]
wmin = torch.min(daily_bikes[:,9,:])
wmax = torch.max(daily_bikes[:,9,:]) - 1
print(wmin,wmax)
daily_bikes[:,9,:] = (daily_bikes[:,9,:] - wmin)/wmax
#* For temp data
temp = daily_bikes[:,10,:]
temp_min = torch.min(temp)
temp_max = torch.max(temp)
temp = (temp - temp_min)/(temp_max - temp_min)
print(temp)
#* We can use mean and standard deviation to transform data
temp = daily_bikes[:,10,:]
temp = (temp - torch.mean(temp))/torch.std(temp)
print(temp)
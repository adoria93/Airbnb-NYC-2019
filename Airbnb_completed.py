#!/usr/bin/env python
# coding: utf-8

# # Airbnb
# When traveling to a destination, one of the first things a person thinks about is lodging. Depending on the location, hotel prices can be fairly expensive, even if a person is only staying overnight. Various websites are devoted to finding the best price for a hotel (trivago, hotels.com, kayak, etc.).
# 
# Airbnb is different. Airbnb is an online marketplace for arranging or offering lodging. The company does not own any of the real estate listings or host any events; it is just the broker, and receives a commission from each booking. The company was conceived after the founders put an air mattress in their living room, turning their apartment into a bed and breakfast in order to offset the high rent cost in San Francisco.
# 
# Since its inception, guests and hosts have used Airbnb to expand on traveling possibilities, presenting a more unique, personalized way of experiencing the world. The idea that someone could offer up an extra room to make some extra cash is very appealing to many people, especially in places where the cost of living is high, such as New York City.
# 
# # How it Works
# 
# Airbnb is a platform for hosts to accommodate guests with short-term lodging and tourism-related activities. You can search using various filters, such as lodging type, location and price. Users must provide payment information before booking and have the ability to chat with the hosts. Hosts provide various information such as prices, event listings, rules, and amenities. Pricing is determined by the host, though Airbnb does recommend prices.
# 
# Advantages compared to hotels:
# * Can be anywhere in the city
# * Wide array of different options to choose from
# * Might be cheaper than a hotel
# 
# Disadvantages compared to hotels:
# * May not be available when you want, hotels typically have many rooms available at any given time
# * Subject to user feedback only, no strength in the "brand"
# * No set rates, hosts charge what they feel like charging
# 
# Hotels typically have a set rate per night (depending on the time of year). In January 2020, hotel rooms in New York City look to go between 100 and 125 dollars per night, based on a quick Google search of the area. This got me thinking, do Airbnb users pay more or less than that per night on average?

# In[5]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

#Importing the data and taking a look at it
airbnb = pd.read_csv("AirBNB_NYC_2019.csv")
airbnb.head()


# # Location
# 
# ![nyc-boroughs-map.jpg](attachment:nyc-boroughs-map.jpg)
# 
# New York City is often referred to collectively as the five boroughs:
# 
# * **Manhattan** - the smallest and most densely populated borough. It is the symbol of New York City, home to many skyscrapers and landmarks (like Times Square and Central Park). Often described as the cultural, financial, media and entertainment capital of the world.
# * **Brooklyn** - the most populous borough. It is known for its cultural, social, and ethic diversity. Brooklyn has evolved into a thriving hub of entrepreneurship and high technology startup firms.
# * **Queens** - the geographically largest borough. It is the most ethnically diverse urban area in the world. Queens is the site of Citi Field where the New York Mets play. In addition to this, John F. Kennedy International Airport and LaGuardia Airport are located in this borough.
# * **The Bronx** - the only borough that is part of the United States mainland. Home to the largest cooperatively owned housing complex in the United States, it is also the location of Yankee Stadium. In addition to this, the world's largest metropolitan zoo, the Bronx Zoo is located here as well as the New York Botanical Garden and Pelham Bay Park (the largest park in New York City)
# * **Staten Island** - the most suburban in character of the five boroughs. Connected to Brooklyn by the Verrazzano-Narrows Bridge and Manhattan via the Staten Island Ferry. Home to the Staten Island Greenbelt, one of the last undisturbed forests in the city.
# 
# No matter where you are staying in New York City, there is always something to do!

# In[3]:


#What neighborhoods are in the data?
neighbourhood_count = airbnb.neighbourhood_group.value_counts()
print(neighbourhood_count)


# In[45]:


#Visualization of the above data
labels = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
plt.pie(neighbourhood_count, labels = labels, autopct='%1.1f%%')

plt.axis('equal')

plt.tight_layout()
plt.show()


# The vast majority of listings on Airbnb are in Manhattan and Brooklyn, making up 85.4% of the total rooms. This makes sense since Manhattan is densly populated and Brooklyn has a large population in general.

# In[46]:


#What types of rooms are available?
room_type_count = airbnb.room_type.value_counts()
print(room_type_count)


# In[47]:


#Visualization
labels = ["Entire home/apt", "Private room", "Shared room"]
plt.pie(room_type_count, labels = labels, autopct='%1.1f%%')

plt.axis('equal')

plt.tight_layout()
plt.show()


# Airbnb offers a wide variety of spaces, all grouped into the following room types:
# 
# * **Entire home/apt** - You get the entire dwelling to yourself, usually including a bedroom, bathroom, kitchen, and a  dedicated entrance. The best option if you are seeking a "home away from home". Probably the best option if you are staying in a location for a few days.
# * **Private room** - You get a private room in the apartment/home for sleeping and might share some spaces with others (either the host or other guests). Good for if you want a little privacy, but value a local connection.
# * **Shared room** - You are sleeping in a space shared with others and share the entire space with other people. Popular among flexible travelers looking for budget-friendly stays or someone looking for new friends.
# 
# For New York City, entire apartment/home makes up (52%) of the available rooms, followed closely by private rooms (45.7%). Very few hosts are offering shared rooms. This surprises me. I didn't expect so many hosts to offer the entire apartment/home to the guest. I figured that people would be using airbnb to help offset the higher cost of living in the city by bringing in some extra money each month by renting out spare rooms in their place.
# 
# # Pricing
# Since people are offering either their entire home or just a room, I wonder if that changes what they are charging per night. If a person is offering the entire dwelling, are they charging more than someone just offering up a spare room?

# In[6]:


#How expensive are airbnb prices in NYC?
airbnb.price.describe()


# In[64]:


#A visualization of the data, with prices greater than $700 a night removed (explained below)
prices = airbnb.drop(airbnb[airbnb.price > 700].index)
plt.hist(prices.price, bins= 20)
plt.title("Airbnb Prices in NYC 2019 (limited)")
plt.xlabel("Price ($)")
plt.ylabel("Number of Available Rooms")
plt.show()


# Looking at the histogram above, it is clear that this data is skewed to the right (mean is larger than the median). Because of this, I will be using the median to help describe where the center of this dataset is, since the median is more resistant to large outliers. For this dataset, the median price for an Airbnb room in NYC in 2019 is $106 per night, which appears to be on par with what hotels are charging.
# 
# It is worth mentioning that the histogram used to visualize the data does not include the full data because I wanted the histogram to at least be useable to some degree. If I include the full data, the histogram won't be useful at all. For comparision, this the histogram for the full dataset:

# In[65]:


plt.hist(airbnb.price, bins= 20)
plt.title("Airbnb Prices in NYC 2019 (full)")
plt.xlabel("Price ($)")
plt.ylabel("Number of Available Rooms")
plt.show()


# As you can see, this histogram is pretty much useless because of the large outliers in this dataset. But these outliers are still useful because they aren't data entry errors, they are actual prices that hosts are charging! To help determine how prices change, I will be breaking up the data based on neighbourhood group as well as room type.

# In[49]:


#Create new datasets for each neighbourhood group (ex. Brooklyn, Manhattan, etc.)
manhattan = airbnb[airbnb["neighbourhood_group"] == "Manhattan"]
brooklyn = airbnb[airbnb["neighbourhood_group"] == "Brooklyn"]
queens = airbnb[airbnb["neighbourhood_group"] == "Queens"]
bronx = airbnb[airbnb["neighbourhood_group"] == "Bronx"]
staten_island = airbnb[airbnb["neighbourhood_group"] == "Staten Island"]

print("Manhattan:")
print(manhattan.price.describe())
print()
print("Brooklyn:")
print(brooklyn.price.describe())
print()
print("Queens:")
print(queens.price.describe())
print()
print("Bronx:")
print(bronx.price.describe())
print()
print("Staten Island:")
print(staten_island.price.describe())


# Ranking for neighbourhoods (most to least expensive), using the median:  Manhattan (150) > Brooklyn (90) > Queens (75) = Staten Island (75) > Bronx (65)
# 
# The majority of the listings are in Manhattan and Brooklyn, so it makes sense that there is more variation in the prices. With all of the stuff to do in New York City, Manhattan would be the first choice for a person to stay, since it would be close to a lot of places. But, if you're willing to spend a bit more time getting to your location, you can save a bit of money by just staying in Brooklyn.

# In[58]:


#Breakdown by room type
room_types = airbnb[["neighbourhood_group", "room_type", "price"]]

room_types.groupby(["neighbourhood_group", "room_type"]).describe()


# The above table is a breakdown of each available room type in each neighbourhood group. I thought this would be interesting to look at to see how much hosts were charging depending on the type of room they were offering. In particular, I was interested in seeing how much more (or less) it would cost, on average, for the entire home/apt vs. just a private room.
# 
# Above I mentioned that Manhattan would be the place to be when it comes to first choices to staying in New York City. The best deal for a Manhattan airbnb room is to get a shared room, which would run you around 69. If you're willing to spend a bit more, you can get a private room for 90.
# 
# If you're willing to travel a bit further, you can get a good deal in Brooklyn with 65 for a private room and 36 for a shared room.
# 
# # Conclusion
# Airbnb is changing up the lodging industry. No longer are you restricted to just staying in a hotel, you can create your own unique experience using this service, wherever you are in the world.

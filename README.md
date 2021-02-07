# Movie-Rating-Prediction-and-Recommendation-System


## This project was carried out by talented students under my guidance (Saifali Gulamabbas Patel, Rinaldi James Michael and	Syed Muddassir Ahmed)

A Recommender System is the most profitable solution for an organization that caters to a number of users having a number of options. As the name suggests, the system uses algorithms to recommend content (videos, advertisements, news etc.) to an individual based on his/her usage. There are several ways to design recommendation systems: \\
    1. Collaborative Filtering \
    2. Content based method \
    3. Association Mining \
Collaborative Filtering uses past user interaction data to suggest new items. The interactions can be placed in a “user-items interaction matrix”. Collaborative Filtering can further be divided into Memory based and Model Based. The advantage of this filtering is that the acquired user and item data is enough to predict similar items to recommend to similar users. Since it only considers past data based on interactions between users and items to make suggestions, it is subjected to the cold start problems that makes it unable to make recommendations to new users or suggest new items to existing users.    

Content based method utilizes more information about user and items to create a model that learns how a user having certain features will prefer a particular item having a set of characteristics over other items. Content based method uses available information about user and items like features of a user or characteristic or content of an item that explain the interaction between the user and item. The method doesn’t suffer from poor recommendations due to insufficient beginner data.

Association mining works by searching relations or correlations among items in a database. An association is a type of condition in the form X->Y Where X and Y are two sets of items. It is used to find the correlation between X and Y. For example, if a user buys item X then how likely will the user buy item Y.

## Dataset 
The dataset is provided by Netflix as a part of the Netflix prize data competition. It includes movies rated by users on a scale from 1 to 5. The competition was to find the best algorithm to increase the predictions and improve the recommendation system.
https://www.kaggle.com/netflix-inc/netflix-prize-data#combined_data_1.txt

Year of movies release.
![alt text](https://github.com/siddhaling/Movie-Rating-Prediction-and-Recommendation-System/blob/main/images/1.jpg)

Movie rating distribution.

![alt text](https://github.com/siddhaling/Movie-Rating-Prediction-and-Recommendation-System/blob/main/images/2.jpg)

Movie rating per day.

![alt text](https://github.com/siddhaling/Movie-Rating-Prediction-and-Recommendation-System/blob/main/images/3.jpg)

Rating per movie.

![alt text](https://github.com/siddhaling/Movie-Rating-Prediction-and-Recommendation-System/blob/main/images/4.jpg)



## Method for recommendation
We first load our Netflix user data then we perform some preprocessing operations to clean our data. We further proceed to explore the data and uncover some interesting results like release year of movies , no of ratings per movies and per user before diving deeper .
We have used four techniques for recommending \
    • Mean Rating \
We can calculate the mean rating for the movies which can be used to recommend movies to all the users even when no past information about them is present.
Drawback of this method is movies with few ratings can have a higher mean.
This can be further extended in such a way that a rating r given by a user u would be close to the average of the ratings given by n users that are similar to u. \
    • Weighted Mean Rating \
We can assign weights to the movies so as to include the factor of number of votes, this addresses the drawback of mean rating.
This can further be extended in such a way that users who are more similar to the target user, their ratings would be considered more than other users who are less similar. \
    • Cosine User-User Similarity \
The ideas discussed above can now be used for user-specific recommendation, the S or the similarity factor that is used to group similar users together can we found out using this method.
The implementation of this for user having index 0 and finding 10 similar users to him and recommending 10 top movies to him.

We use Cross-Validation techniques along with the techniques to validate our models using multiple folds.
## Results
First we will look at the results obtained from exploring the data initially :
Release Year of movies More number of movies released towards 2000’s.The declining curve after 2000 is because of incomplete data.

![alt text](https://github.com/siddhaling/Movie-Rating-Prediction-and-Recommendation-System/blob/main/images/9.jpg)

We now move onto look at the results after applying the algo’s. After applying Mean-Rating we get the following result, Root Mean Square Error as = 0.9938

Ranking of top 10 movies

![alt text](https://github.com/siddhaling/Movie-Rating-Prediction-and-Recommendation-System/blob/main/images/10.jpg)

If we use Weighted-Mean Rating we get, RMSE as 0.9947 this is higher than the Mean rating as this represents a clearer picture because it considers the number of ratings too.

![alt text](https://github.com/siddhaling/Movie-Rating-Prediction-and-Recommendation-System/blob/main/images/11.jpg)

Cosine User-User Similarity gives us We take the first user and find 10 similar users who are similar to him and based on that data we find top 10 movies that he would like.
After computing this we get RMSE of 1.3357

![alt text](https://github.com/siddhaling/Movie-Rating-Prediction-and-Recommendation-System/blob/main/images/12.jpg)

 The cross validation was performed on a sample of the data set.
 
 ![alt text](https://github.com/siddhaling/Movie-Rating-Prediction-and-Recommendation-System/blob/main/images/13.jpg)
 
 # Further Projects and Contact
www.researchreader.com

https://medium.com/@dr.siddhaling

Dr. Siddhaling Urolagin,\
PhD, Post-Doc, Machine Learning and Data Science Expert,\
Passionate Researcher, Deep Learning, Machine Learning and applications,\
dr.siddhaling@gmail.com

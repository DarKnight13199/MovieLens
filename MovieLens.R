
# 1) Downloading and creating dataset----

# Following code was provided by the HarvardX to download the files and turn it into data set. 

# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)

# 2) Exploring Datasets----

#Loading Libraries
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")

library(ggplot2)
library(dplyr)
library(ModelMetrics)

#printing first five rows of the Edx dataset.
head(edx,  n=5)

summary(edx)

print(paste0('We have ', n_distinct(edx$userId), ' distinct users, ',
             n_distinct(edx$movieId), ' movies and ',
             n_distinct(edx$genres), ' genres in the dataset'))


# histogram of ratings
ggplot(edx, aes(x=rating)) + geom_histogram(fill="lightblue", color="black") + 
  labs(x="Ratings", y="Count") +
  ggtitle("Ratings Distribution")


# Separate genres into separate rows
edx_split <- edx %>% 
  separate_rows(genres, sep = "\\|")

# Count the occurrences of each genre
genre_counts <- edx_split %>% 
  count(genres)

# Sort by number of occurrences and display the top 10 genres
top_genres <- genre_counts %>% 
  arrange(desc(n)) %>% 
  top_n(10)

print(top_genres)

#plotting genres based on ratings
ggplot(top_genres, aes(x=n, y=reorder(genres,n)))+
  geom_bar(stat='identity', fill="steelblue", width = 0.5)+ 
  labs(x="", y="Number of ratings", title="Top 10 genres based \n on ratings") +
  geom_text(aes(label= n), hjust=-0.1, size=3)

# Count the occurrences of each movie
movie_counts <- edx %>% 
  count(title)

# Sort by number of occurrences and display the top 10 movies
top_movies <- movie_counts %>% 
  arrange(desc(n)) %>% 
  top_n(10)

print(top_movies)

# 3) Data Preprocessing----

#creating a dataset that will be used for modeling. 
df <- edx[,colnames(edx)!="timestamp"]
df$year_released <- gsub(".*\\(([0-9]{4})\\).*", "\\1", df$title)


# 4) Regression----
# calculate the average of all ratings of the df data set
mu <- mean(df$rating)

# Calculate b_i and b_u
b_i <- df %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))

b_u <- df %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

# Predict ratings
predicted_ratings_bu <- df %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Calculate RMSE
RMSE(predicted_ratings_bu, edx$rating)

# 5) Regularization----

#tuning parameters
lambdas <- seq(0, 5, 0.25)

#RMSES Table with each tuning parameter
rmses <- sapply(lambdas, function(l){
  
  #Mean Ratings
  mu_reg <- mean(df$rating)
  
  #beta based on movieID
  b_i_reg <- df %>% 
    group_by(movieId) %>%
    summarize(b_i_reg = sum(rating - mu_reg)/(n()+l))
  
  #beta based on userid and movieId    
  b_u_reg <- df %>% 
    left_join(b_i_reg, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u_reg = sum(rating - b_i_reg - mu_reg)/(n()+l))
  
  #predicting the model based on calculated betas.
  predicted_ratings_b_i_u <- 
    df %>% 
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    mutate(pred = mu_reg + b_i_reg + b_u_reg) %>%
    .$pred
  
  
  return(RMSE(predicted_ratings_b_i_u, edx$rating)) 
})

plot(y =rmses, x =lambdas)

#Tuning parameters and RMSE table

tpr <- data.frame(Lambdas = lambdas,
                  RMSE = rmses)

tpr[which.min(tpr$RMSE),]
# MODEL TRAINING SETUP --------------------

# load the necessary libraries
library(dplyr)
library(readr)
library(stringr)
library(purrr)
library(tidyr)
library(keras) # use install_keras() if running for the first time. See https://keras.rstudio.com/ for details
np <- reticulate::import("numpy", convert=FALSE) # needed for a small amount of nparray manipulation
ku <- reticulate::import("keras.utils", convert=TRUE)
k <- reticulate::import("keras", convert=TRUE)
tf <- reticulate::import("tensorflow")
# use_virtualenv("r-reticulate")
# load all of the parameters. They are stored in a separate file so they can be used when
# running the model too
source("parameters.R")


# load the data. We don't need most of the columns in it, and we need to clean the strings.
pet_data <- 
  read_csv("seattle_pet_licenses.csv", 
           col_types = cols_only(`Animal's Name` = col_character(),
             Species = col_character(),
             `Primary Breed` = col_character(),
             `Secondary Breed` = col_character())) %>%
  rename(name = `Animal's Name`,
         species = `Species`,
         primary_breed = `Primary Breed`,
         secondary_breed = `Secondary Breed`) %>%
  mutate_all(toupper) %>%
  filter(!is.na(name),!is.na(species)) %>% # remove any missing a name or species
  filter(!str_detect(name,"[^ \\.-[a-zA-Z]]")) %>% # remove names with weird characters
  mutate_all(stringi::stri_trans_tolower) %>%
  filter(name != "") %>%
  mutate(id = row_number())


# modify the data so it's ready for a model
# first we add a character to signify the end of the name ("+")
# then we need to expand each name into subsequences (S, SP, SPO, SPOT) so we can predict each next character.
# finally we make them sequences of the same length. So they can form a matrix
# one = pet_data[1,]
# l = c( "t", "i" ,"n" ,"k" ,"e" ,"r" ,"d" ,"e", "l", "l", "e", "+")
# the subsequence data
subsequence_data <-
  pet_data %>%
  mutate(accumulated_name =
           name %>%
           str_c("+") %>% # add a stop character
           str_split("") %>% # split into characters
           # accumulate uses previous value (iteration). c is a base function
           map( ~ purrr::accumulate(.x,c)) # make into cumulative sequences.
         ) %>%
  select(accumulated_name) %>% # get only the column with the names
  unnest(accumulated_name) %>% # break the cumulations into individual rows
  arrange(runif(n())) %>% # shuffle for good measure
  pull(accumulated_name) # change to a list

# the name data as a matrix. This will then have the last character split off to be the y data
# this is nowhere near the fastest code that does what we need to, but it's easy to read so who cares?
# x = subsequence_data[1]
# TF_ENABLE_ONEDNN_OPTS=0
text_matrix <-
  subsequence_data %>%
  map(~ character_lookup$character_id[match(.x,character_lookup$character)]) %>% # change characters into the right numbers
  pad_sequences(maxlen = max_length+1)# %>% # add padding so all of the sequences have the same length
  # to_categorical(num_classes = as.integer(num_characters)) # 1-hot encode them (so like make 2 into [0,1,0,...,0])
text_matrix = ku$to_categorical(reticulate::np_array(text_matrix), num_classes = as.integer(num_characters))

x_name <- np$delete(text_matrix, as.integer(max_length), 1L) # make the X data of the letters before
y_name <- np$delete(text_matrix, as.integer((1:max_length)-1), 1L)$squeeze() # make the Y data of the next letter


# CREATING THE MODEL ---------------

# the input to the network
input <- layer_input(shape = c(max_length,as.integer(num_characters))) 

# the name data needs to be processed using an LSTM, 
# Check out Deep Learning with R (Chollet & Allaire, 2018) to learn more.
# if we were using words instead of characters, or we had 10x the datapoints,
# we'd want to use more lstm layers instead of just two
output <- 
  input %>%
  layer_lstm(units = 32L, return_sequences = TRUE) %>%
  layer_lstm(units = 32L, return_sequences = FALSE) %>%
  layer_dropout(rate = 0.2) %>% # keep from overfitting
  layer_dense(as.integer(num_characters)) %>% # 31 char
  layer_activation("softmax") # to make probabilities

# the actual model, compiled
model <- keras_model(inputs = input, outputs = output) |> 
  compile(
    loss = 'categorical_crossentropy',
    optimizer = "adam" # doesn't super matter
  )
model = tf$keras$Model(inputs=input, outputs=output)
model = tf$keras$models$Model(inputs=input, outputs=output)
# keras::compile(model)
# x = k$models$Model$compile(model, optimizer="Adam")
model$compile
mode$fit(y_name, x_name)
x=k$Model$compile
k$Model$fit(x)
# reticulate::py_eval(tf.keras.Model(input, output))
# RUNNING THE MODEL ----------------

# here we run the model through the data 25 times. 
# In theory the more runs the better the results, but the returns diminish
fit_results <- model %>% fit(
  x_name, 
  y_name,
  batch_size = 64, # model learns better when this is smaller but also means there's more data to go through
  epochs = 25 # how long it takes to get optimal results
)

# SAVE THE MODEL ---------------

# save the model so that it can be used in the future
save_model_hdf5(model,"model.h5")

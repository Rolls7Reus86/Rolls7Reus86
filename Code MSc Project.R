#import required libraries
library(plyr)
library(stringr)
library(e1071)
library(RTextTools)
library(tm)
library(NLP)
library(party)
#Loading the Word Polarity List and editing for simplification
afinn_list = read.delim(file='AFINN-111.txt', header=FALSE, stringsAsFactors=FALSE)
names(afinn_list) = c('word', 'score')
afinn_list$word = tolower(afinn_list$word)
vNegTerms = afinn_list$word[afinn_list$score==-5 | afinn_list$score==-4 |
                              afinn_list$score==-3]

negTerms <- c(afinn_list$word[afinn_list$score==-2 | afinn_list$score==-1], "second-
rate", "moronic", "third-rate", "flawed", "juvenile", "boring", "distasteful",
              
              "ordinary", "disgusting", "senseless", "static", "brutal", "confused", "disappointing",
              "bloody", "silly", "tired", "predictable", "stupid", "uninteresting", "trite",
              "uneven", "outdated", "dreadful", "bland")
posTerms <- c(afinn_list$word[afinn_list$score==2 | afinn_list$score==1], "first-rate",
              "insightful", "clever", "charming", "comical", "charismatic", "enjoyable", "absorbing",
              "sensitive", "intriguing", "powerful", "pleasant", "surprising", "thought-provoking",
              "imaginative", "unpretentious")
vPosTerms <- c(afinn_list$word[afinn_list$score==5 | afinn_list$score==4 |
                                 afinn_list$score==3], "uproarious", "riveting", "fascinating", "dazzling", "legendary")
#Loading the positive and negative reviews
posText <- read.delim(file='game_p.txt', header=FALSE, stringsAsFactors=FALSE)
posText <- posText$V1
posText <- unlist(lapply(posText, function(x) { str_split(x, "\n") }))
negText <- read.delim(file='game_n.txt', header=FALSE, stringsAsFactors=FALSE,
                      quote="")
negText <- negText$V1
negText <- unlist(lapply(negText, function(x) { str_split(x, "\n") }))
text_all=c(posText, negText);head(text_all)
#Text Cleaning and Word Cloud
#Extracting text from the data frame into new corpus
data_2=Corpus(VectorSource(text_all))
#convert to lower case
data_2=tm_map(data_2, content_transformer(tolower))
#remove emojis (replace with a space)
data_2<-tm_map(data_2, content_transformer(gsub), pattern="\\W",replace=" ")
#remove URLs
remove_URL=function(x) gsub("http[^[:space:]]*", "", x)
data_2=tm_map(data_2, content_transformer(remove_URL))
#remove anything other than English letters or space
remove_non_eng=function(x) gsub("[^[:alpha:][:space:]]*", "", x)
data_2=tm_map(data_2, content_transformer(remove_non_eng))
#remove stopwords
data_2=tm_map(data_2, removeWords, stopwords("english"))
#remove extra whitespace
data_2=tm_map(data_2, stripWhitespace)
#remove numbers
data_2=tm_map(data_2, removeNumbers)
#remove additional words in case it is needed to
#data_2=tm_map(data_2, removeWords, c("WORD1", "WORD2", ... , "WORDn"))
#remove punctuations
data_2=tm_map(data_2, removePunctuation)
#keep a copy
Train_data_Copy=data_2
#Analysis of the data
#Build a term document matrix (to get word/term frequencies)
dtm=TermDocumentMatrix(data_2)
m=as.matrix(dtm)
v=sort(rowSums(m),decreasing=TRUE)
d=data.frame(word = names(v),freq=v)
head(d, 20)
#Wordcloud for visualizing the data
library(RColorBrewer)
library(wordcloud)
set.seed(7)
wordcloud(words = d$word, freq = d$freq, min.freq = 10,max.words=100,
          random.order=F,scale = c(3, 0.5), colors = rainbow(8))
#Function to calculate number of words in each category within a sentence
sentimentScore <- function(sentences, vNegTerms, negTerms, posTerms, vPosTerms){
  final_scores <- matrix('', 0, 5)
  scores <- laply(sentences, function(sentence, vNegTerms, negTerms, posTerms,
                                      vPosTerms){
    initial_sentence <- sentence
    #remove unnecessary characters and split up by word
    sentence <- gsub('[[:punct:]]', '', sentence)
    sentence <- gsub('[[:cntrl:]]', '', sentence)
    sentence <- gsub('\\d+', '', sentence)
    sentence <- tolower(sentence)
    wordList <- str_split(sentence, '\\s+')
    words <- unlist(wordList)
    #Build vector with matches between sentence and each category
    vPosMatches <- match(words, vPosTerms)
    posMatches <- match(words, posTerms)
    vNegMatches <- match(words, vNegTerms)
    negMatches <- match(words, negTerms)
    #Sum up number of words in each category
    vPosMatches <- sum(!is.na(vPosMatches))
    posMatches <- sum(!is.na(posMatches))
    vNegMatches <- sum(!is.na(vNegMatches))
    negMatches <- sum(!is.na(negMatches))
    score <- c(vNegMatches, negMatches, posMatches, vPosMatches)
    #add row to scores table
    newrow <- c(initial_sentence, score)
    final_scores <- rbind(final_scores, newrow)
    return(final_scores)
  }, vNegTerms, negTerms, posTerms, vPosTerms)
  return(scores)
}
#Build tables of positive and negative sentences with scores
posResult <- as.data.frame(sentimentScore(posText, vNegTerms, negTerms, posTerms,
                                          vPosTerms))
negResult <- as.data.frame(sentimentScore(negText, vNegTerms, negTerms, posTerms,
                                          vPosTerms))
posResult <- cbind(posResult, 'positive')
colnames(posResult) <- c('sentence', 'vNeg', 'neg', 'pos', 'vPos', 'sentiment')
negResult <- cbind(negResult, 'negative')
colnames(negResult) <- c('sentence', 'vNeg', 'neg', 'pos', 'vPos', 'sentiment')
#Combine the positive and negative tables
results <- rbind(posResult, negResult)
results[c(1,2,124:127,249,250),]
results$sentiment[2]
#Using naïve Bayes Classifier
classifier <- naiveBayes(results[,2:5], results[,6])
head(classifier)
#Display the confusion table for the classification run on the same data
confTable <- table(predict(classifier, results), results[,6],
                   dnn=list('predicted','actual'))
confTable
#Run a binomial test for confidence interval of results
binom.test(confTable[1,1] + confTable[2,2], nrow(results), p=0.5)
#Using SVM for classification
model_svm <- svm(results$sentiment ~ . , results);
predictedY <- predict(model_svm, results)
confTable2<- table(predictedY, results[,6], dnn=list('predicted','actual'))
confTable2
#run a binomial test for confidence interval of results
binom.test(confTable2[1,1] + confTable2[2,2], nrow(results), p=0.5)
#Using Binary Classification Tree
model_tree = ctree(results$sentiment ~ ., results);plot(model_tree)
predictedTree <- predict(model_tree, results)
confTable3<- table(predictedTree, results[,6], dnn=list('predicted','actual'))
confTable3
#run a binomial test for confidence interval of results
binom.test(confTable3[1,1] + confTable3[2,2], nrow(results), p=0.5)
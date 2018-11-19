# remove object
rm(v)

# return elements in vector except the second
v=c(1:10)
v[-2]

# uniform distribution
x=runif(10)
# norm distribution
y=rnorm(10)

# plot 2 plots in a figure
par(mfrow=c(2,1))
plot(x, y)
plot(y, x)

# list objects in the environment
ls()

# read csv
df=read.csv(filename)

# dataframe operation
names(df)      # name of variables
dim(df)        # dimension of data
summary(df)    # summary of data
df$var         # column var
attach(data)   # create a workspace with all named variable in df and variables in the current workspace 
search()       # return a list of objects and attached packages

# clear the environment
rm(list=ls())
# cleare plots
graphics.off()
dev.off()
# clear the console
cat("\014")


# create a table of the counts at each combination of factor levels
t = table(df$var)
# express table entries as fraction of marginal table 
prop = prop.table(t)
# round the values
pf = round(prop, 1)
# concatenate vectors after converting to character
paste(1:12, c('st', 'nd', 'rd', rep('th', 9)), sep="")

# produce the five number summary: Min, Q1, Median, Q3, Max
summary(df$var)
# Obtain inter-quartile range
IQR(df$var)

# boxplot example for table where row and column correspond to a variable, respectively
# in this example, the column corresponds to college and the row to the corresponding graduation rates
boxplot(grad_data, xlab='Colleges', ylab='Graduation Rates', main="Comparison of Graduation Rates")
# boxplot example for two-column table where each column represents a variable
# in this example, the College column displays college name and the GradRate column shows the corresponding graduation rates
boxplot(grad_data2$GradRate~grad_data2$College)

# apply a function to a dataframe
sapply(dataframe, sd) # compute the standard deviation for each column in dataframe

# label data in scatterplot and draw legend
plot(h$height, h$weight, xlab='Height (inches)', ylab='Weight (lbs)', col='blue')
points(h$height[h$gender==1], h$weight[h$gender==1], col='red')
legend(55, 220, pch = 1, col = c('red', 'blue'), legend = c('females', 'males'))

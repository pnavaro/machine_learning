using Plots, CSV
include("utils.jl")
include("lasso.jl")
# Load temperature data
data = CSV.read(download("https://raw.githubusercontent.com/eriklindernoren/ML-From-Scratch/master/mlfromscratch/data/TempLinkoping2016.txt"), delim="\t");

time = reshape(collect(data.time), size(data)[1], 1)
temp = collect(data.temp);

X = time # fraction of the year [0, 1]
y = temp;

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

poly_degree = 13

model = LassoRegression(degree=15, 
                        reg_factor=0.05,
                        n_features=1, 
                        learning_rate=0.001,
                        n_iterations=4000)
size(X_train)



fit(model, X_train, y_train)

# +
X = collect(reshape(1:12, 3, 4))

for j in 1:size(X)[2]
    X[1:end,j] = normalize(vec(X[1:end,1]))
end
# -

# Training error plot
n = len(model.training_errors)
training, = plt.plot(range(n), model.training_errors, label="Training Error")
plt.legend(handles=[training])
plt.title("Error Plot")
plt.ylabel('Mean Squared Error')
plt.xlabel('Iterations')
plt.show()

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print ("Mean squared error: %s (given by reg. factor: %s)" % (mse, 0.05))

y_pred_line = model.predict(X)

# Color map
cmap = plt.get_cmap('viridis')

# Plot the results
m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
plt.plot(366 * X, y_pred_line, color='black', linewidth=2, label="Prediction")
plt.suptitle("Lasso Regression")
plt.title("MSE: %.2f" % mse, fontsize=10)
plt.xlabel('Day')
plt.ylabel('Temperature in Celcius')
plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
plt.show()


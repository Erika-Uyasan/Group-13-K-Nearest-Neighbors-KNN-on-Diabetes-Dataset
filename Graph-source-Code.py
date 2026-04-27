import matplotlib.pyplot as plt
# K values and corresponding accuracies 
k_values = [3, 5, 7, 9]
accuracies = [0.7208, 0.7468, 0.7468, 0.7143]
# plot
plt.figure()
plt.plot(k_values, accuracies, marker='o')
# Labels and title
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K in KNN")

# grid
plt.grid()

# save the figure 
plt.savefig("accuracy_vs_k.png")

# show graph
plt.show()

import lejepa

# Choose a univariate test (Epps-Pulley in this example)
univariate_test = lejepa.univariate.EppsPulley(num_points=17)

# Create the multivariate slicing test
loss_fn = lejepa.multivariate.SlicingUnivariateTest(
    univariate_test=univariate_test,
    num_slices=1024
)

# Compute the loss (embeddings: [num_samples, num_dims])
loss = loss_fn(embeddings)
loss.backward()
import torch

x_aug = torch.randn(5, 5)

#print(x_aug)
#print(x_aug[0].shape)

# unsqueeze(1) gives a col vector of d x 1. unsqueeze(0) gives a row vector of 1 x d.
# print(x_aug[0].unsqueeze(1) @ x_aug[0].unsqueeze(0))

batch_size, dims = x_aug.shape[0], x_aug.shape[1]
# print(batch_size, dims)
mean = torch.sum(x_aug, dim = 0) / batch_size 
# print(mean)
sum = torch.empty(dims, dims) 
# print(sum)
for i in range(0, batch_size):
    sum = sum + ((x_aug[i] - mean).unsqueeze(1) @ (x_aug[i] - mean).unsqueeze(0)) # (3) in paper

covar = sum / (batch_size - 1)
print(covar)
print(torch.cov(x_aug.t()))

print(torch.sum(torch.square(torch.tril(torch.cov(x_aug.t()), diagonal = -1))))

print(torch.equal(covar, torch.cov(x_aug.t()))) # Says false, but probably because of minor differences. 
                                                # when you print though, you see that they are equal.

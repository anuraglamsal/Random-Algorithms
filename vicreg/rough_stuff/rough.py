import torch

"""x = torch.randn(3, 5) # col x row

print(x)
print(x.shape)

print(torch.sum(x, dim = 0)) # typical vector addition

print(x.t()[0]) # .t() does transpose
print(x.t().shape[0])

m_1 = torch.randn(1, 5) # (col) vector of size 5
m_2 = torch.randn(1, 5) # same as above 

print(m_1.t() @ m_2) # 5 x 5
print(m_1 @ m_2.t()) # 1 x 1

t_1 = torch.randn(2, 5, 5)

print(t_1)
print(torch.sum(t_1, dim = 0)) # 5 x 5 

q = torch.randn(2, 1, 5)

print(q)
print(q.permute(2, 1, 0)) # transposes

sum = 0 
g = torch.randn(1, 5)
q = torch.randn(1, 5)
sum = sum + g
sum = sum + q
# yeah, this way of summing works..
print(g, q, sum)

h = torch.randn(2, 1, 5)
print(h)
print(h[0][0]) # yeah, works as is intuitive..


h = torch.randn(5, 5) # way more sensible that you get something like this than below..
print(h)
i = torch.randn(5, 1, 5)
print(i)"""

x = torch.empty(2)
x[0] = torch.rand(4)
x[1] = torch.rand(4)
print(x)

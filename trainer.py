#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train script

@author: javad
"""


import torch
 
 
def train(model, X, y, steps=1000, lr=0.05, print_every=200):
    # train the model by minimizing the loss (the negative ELBO)
 
    # the optimizer updates the model's numbers using the gradients
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
    # keep the loss at each step so we can look at it later
    loss_history = []
 
    for step in range(steps):
        # 1. clear old gradients from the last step
        optimizer.zero_grad()
 
        # 2. compute the loss
        loss = model.get_loss(X, y)
 
        # 3. compute the gradients (how to change each number)
        loss.backward()
 
        # 4. take one step to update the numbers
        optimizer.step()
 
        # save the loss value
        loss_history.append(loss.item())
 
        # print progress once in a while
        if print_every > 0 and step % print_every == 0:
            print(f"step {step:4d}   loss = {loss.item():.2f}")
 
    return loss_history
 
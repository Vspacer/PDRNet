import torch
import torch.nn as nn
import cv2
import numpy as np
import os, glob, datetime, time
import torch.nn.init as init 





def dwt_97_hor_lifting(X, n_taps, taps):
    [n,c,h,w] = X.shape
    Z1 = torch.zeros([n, c, h, 2], dtype=torch.float32, requires_grad=True)
    Z2 = torch.zeros([n, c, h, 1], dtype=torch.float32, requires_grad=True)
    #extended for boundary extension
    T = torch.cat([Z1.cuda(), X, Z2.cuda()], 3)
    for i in range(n_taps):
        # boundary extension
        T[:,:,:,1] = T[:,:,:,3]
        T[:,:,:,w+2] = T[:,:,:,w]
        #lifting phase
        start = 2 + (i+1)%2
        #lifting steps
        #for pos in range(start, w+2, 2):
        #   T[:,:,:,pos] = T[:,:,:,pos] + taps[i]*(T[:,:,:,pos-1] + T[:,:,:,pos+1])
        T[:,:,:,start:w+2:2] = T[:,:,:,start:w+2:2] + taps[i]*(T[:,:,:,start-1:w+1:2] + T[:,:,:,start+1:w+3:2])
    #remove boundary
    Y_tmp = T[:,:,:,2:w+2]
    # separate into low - and high- bands
    Y = torch.cat((Y_tmp[:,:,:,0:w:2]*taps[n_taps], Y_tmp[:,:,:,1:w:2]/taps[n_taps]), 3)
    return Y

def dwt_97_ver_lifting(X, n_taps, taps):
    [n,c,h,w] = X.shape
    Z1 = torch.zeros([n, c, 2, w], dtype=torch.float32, requires_grad=True)
    Z2 = torch.zeros([n, c, 1, w], dtype=torch.float32, requires_grad=True)
    #extended for boundary extension
    T = torch.cat([Z1.cuda(), X, Z2.cuda()], 2)
    for i in range(n_taps):
        # boundary extension
        T[:,:,1,:] = T[:,:,3,:]
        T[:,:,h+2,:] = T[:,:,h,:]
        #lifting phase
        start = 2 + (i+1)%2
        #lifting steps
        #for pos in range(start, h+2, 2):
        #    T[:,:,pos,:] = T[:,:,pos,:] + taps[i]*(T[:,:,pos-1,:] + T[:,:,pos+1,:])
        T[:,:,start:h+2:2,:] = T[:,:,start:h+2:2,:] + taps[i]*(T[:,:,start-1:h+1:2,:] + T[:,:,start+1:h+3:2,:])
    #remove boundary
    Y_tmp = T[:,:,2:h+2,:]
    # separate into low - and high- bands
    Y = torch.cat((Y_tmp[:,:,0:h:2,:]*taps[n_taps], Y_tmp[:,:,1:h:2,:]/taps[n_taps]), 2)
    return Y

def idwt_97_hor_lifting(X, n_taps, taps):
    [n,c,h,w] = X.shape
    Y_tmp = torch.zeros(X.shape, dtype=torch.float32, requires_grad=True)
    Y = Y_tmp.clone()
    # interleave low- and high- band
    Y[:,:,:,0:w:2] = X[:,:,:,:w//2]/taps[n_taps] 
    Y[:,:,:,1:w:2] = X[:,:,:,w//2:w]*taps[n_taps]
    # extended for boundary extension
    Z1 = torch.zeros([n, c, h, 2], dtype=torch.float32, requires_grad=True)
    Z2 = torch.zeros([n, c, h, 1], dtype=torch.float32, requires_grad=True)
    T = torch.cat((Z1.cuda(), Y.cuda(), Z2.cuda()), 3)
    for i in range(n_taps-1, -1, -1):
        #boundary extension
        T[:,:,:,1] = T[:,:,:,3]
        T[:,:,:,w+2] = T[:,:,:,w]
        #lifting phase 
        start = 2 + (i+1)%2
        #lifting steps
        #for pos in range(start, w+2, 2):
        #    T[:,:,:,pos] = T[:,:,:,pos] - taps[i]*(T[:,:,:,pos-1] + T[:,:,:,pos+1])
        T[:,:,:,start:w+2:2] = T[:,:,:,start:w+2:2] - taps[i]*(T[:,:,:,start-1:w+1:2] + T[:,:,:,start+1:w+3:2])
    #remove boundary extension
    Y = T[:,:,:,2:w+2]
    return Y


def idwt_97_ver_lifting(X, n_taps, taps):
    [n,c,h,w] = X.shape
    Y_tmp = torch.zeros(X.shape, dtype=torch.float32, requires_grad=True)
    Y = Y_tmp.clone()
    # interleave low- and high- band
    Y[:,:,0:h:2,:] = X[:,:,:h//2,:]/taps[n_taps]
    Y[:,:,1:h:2,:] = X[:,:,h//2:h,:]*taps[n_taps]
    # extended for boundary extension
    Z1 = torch.zeros([n, c, 2, w], dtype=torch.float32, requires_grad=True)
    Z2 = torch.zeros([n, c, 1, w], dtype=torch.float32, requires_grad=True)
    T = torch.cat((Z1.cuda(),Y.cuda(),Z2.cuda()), 2)
    for i in range(n_taps-1, -1, -1):
        #boundary extension
        T[:,:,1,:] = T[:,:,3,:]
        T[:,:,h+2,:] = T[:,:,h,:]
        #lifting phase 
        start = 2 + (i+1)%2
        #lifting steps
        #for pos in range(start, h+2, 2):
        #    T[:,:,pos,:] = T[:,:,pos,:] - taps[i]*(T[:,:,pos-1,:] + T[:,:,pos+1,:]) 
        T[:,:,start:h+2:2,:] = T[:,:,start:h+2:2,:] - taps[i]*(T[:,:,start-1:h+1:2,:] + T[:,:,start+1:h+3:2,:])
    #remove boundary extension
    Y = T[:,:,2:h+2,:]
    return Y


def dwt_97(X):
    n_taps = 4
    [n,c,h,w] = X.shape
    taps = torch.tensor([-1.586134342, -0.05298011854, 0.8829110762, 0.4435068522, 1.149604398]).cuda()
    X1 = dwt_97_hor_lifting(X, n_taps, taps)
    Y = dwt_97_ver_lifting(X1, n_taps, taps)
    Y1 = Y[:,:,:h//2,:w//2]
    Y2 = Y[:,:,:h//2,w//2:w]
    Y3 = Y[:,:,h//2:h,:w//2]
    Y4 = Y[:,:,h//2:h,w//2:w]
    #Y = np.concatenate((Y1, Y2, Y3, Y4), axis=1)
    Y = torch.cat((Y1,Y2,Y3,Y4), 1)
    return Y

def idwt_97(X):
    n_taps = 4
    taps = torch.tensor([-1.586134342, -0.05298011854, 0.8829110762, 0.4435068522, 1.149604398]).cuda()
    [n,c,h,w] = X.shape
    h = 2*h
    w = 2*w
    c = c//4
    X_tmp = torch.zeros((n, c, h, w), dtype=torch.float32, requires_grad=True).cuda()
    X_tmp[:,:,:h//2,:w//2] = X[:,:c,:,:]
    X_tmp[:,:,:h//2,w//2:w] = X[:,c:2*c,:,:]
    X_tmp[:,:,h//2:h,:w//2] = X[:,2*c:3*c,:,:]
    X_tmp[:,:,h//2:h,w//2:w] = X[:,3*c:4*c,:,:]
    X1 = idwt_97_hor_lifting(X_tmp, n_taps, taps)
    Y = idwt_97_ver_lifting(X1, n_taps, taps)
    return Y

if __name__ == '__main__':
    I = cv2.imread('1.jpg')
    I = I[:,:,0]
    I1 = I
    #print(I)
    #cv2.imshow('I', I)
    I = I.reshape([1,1,I.shape[0], I.shape[1]])
    X = dwt_97(I)
    #cv2.imshow('X', np.clip(X[0,0,:,:], 0, 255).astype(np.uint8))
    #cv2.waitKey(0)
    I = idwt_97(X)
    #I = np.clip(I, 0, 255)
    I = np.around(I)
    I = I.astype(np.uint8)
    I2 = I[0,0,:,:]
    tmp = I1 - I2
    #print(I2)
    #print(I1)
    #cv2.imshow('Tmp', tmp.astype(np.uint8))
    print(np.max(np.max(tmp)))
    print(np.min(np.min(tmp)))
    #print(tmp)
    #cv2.imshow('X', I[0,0,:,:])
    #cv2.waitKey(0)


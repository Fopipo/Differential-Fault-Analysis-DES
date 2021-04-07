# -*- coding: utf-8 -*-

from itertools import product

E = [
    32, 1,  2,  3,  4,  5,
    4,  5,  6,  7,  8,  9,
    8,  9,  10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32, 1
]

PC1 = [
    57, 49, 41, 33, 25, 17, 9,
    1,  58, 50, 42, 34, 26, 18,
    10, 2,  59, 51, 43, 35, 27,
    19, 11, 3,  60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,
    7,  62, 54, 46, 38, 30, 22,
    14, 6,  61, 53, 45, 37, 29,
    21, 13, 5,  28, 20, 12, 4
]

PC2 = [
    14, 17, 11, 24, 1,  5,
    3,  28, 15, 6,  21, 10,
    23, 19, 12, 4,  26, 8,
    16, 7,  27, 20, 13, 2,
    41, 52, 31, 37, 47, 55,
    30, 40, 51, 45, 33, 48,
    44, 49, 39, 56, 34, 53,
    46, 42, 50, 36, 29, 32
]

PC_1 = [
    8,  16, 24, 56, 52, 44,
    36, 0,  7,  15, 23, 55,
    51, 43, 35, 0,  6,  14,
    22, 54, 50, 42, 34, 0,
    5,  13, 21, 53, 49, 41,
    33, 0,  4,  12, 20, 28,
    48, 40, 32, 0,  3,  11,
    19, 27, 47, 39, 31, 0,
    2,  10, 18, 26, 46, 38,
    30, 0,  1,  9,  17, 25,
    45, 37, 29, 0
]


PC_2 = [
    5,  24, 7,  16, 6,  10,
    20, 18, 0,  12, 3,  15,
    23, 1,  9,  19, 2,  0,
    14, 22, 11, 0,  13, 4,
    0,  17, 21, 8,  47, 31,
    27, 48, 35, 41, 0,  46,
    28, 0,  39, 32, 25, 44,
    0,  37, 34, 43, 29, 36,
    38, 45, 33, 26, 42, 0,
    30,  40
]

Sbox = [
    [
        [14,    4,  13, 1,  2,  15, 11, 8,  3,  10, 6,  12, 5,  9,  0,  7],
        [0, 15, 7,  4,  14, 2,  13, 1,  10, 6,  12, 11, 9,  5,  3,  8],
        [4, 1,  14, 8,  13, 6,  2,  11, 15, 12, 9,  7,  3,  10, 5,  0],
        [15,    12, 8,  2,  4,  9,  1,  7,  5,  11, 3,  14, 10, 0,  6,  13]
    ],
    [
        [15,    1,  8,  14, 6,  11, 3,  4,  9,  7,  2,  13, 12, 0,  5,  10],
        [3, 13, 4,  7,  15, 2,  8,  14, 12, 0,  1,  10, 6,  9,  11, 5],
        [0, 14, 7,  11, 10, 4,  13, 1,  5,  8,  12, 6,  9,  3,  2,  15],
        [13,    8,  10, 1,  3,  15, 4,  2,  11, 6,  7,  12, 0,  5,  14, 9]
    ],
    [
        [10,    0,  9,  14, 6,  3,  15, 5,  1,  13, 12, 7,  11, 4,  2,  8],
        [13,    7,  0,  9,  3,  4,  6,  10, 2,  8,  5,  14, 12, 11, 15, 1],
        [13,    6,  4,  9,  8,  15, 3,  0,  11, 1,  2,  12, 5,  10, 14, 7],
        [1, 10, 13, 0,  6,  9,  8,  7,  4,  15, 14, 3,  11, 5,  2,  12]
    ],
    [
        [7, 13, 14, 3,  0,  6,  9,  10, 1,  2,  8,  5,  11, 12, 4,  15],
        [13,    8,  11, 5,  6,  15, 0,  3,  4,  7,  2,  12, 1,  10, 14, 9],
        [10,    6,  9,  0,  12, 11, 7,  13, 15, 1,  3,  14, 5,  2,  8,  4],
        [3, 15, 0,  6,  10, 1,  13, 8,  9,  4,  5,  11, 12, 7,  2,  14]
    ],
    [
        [2, 12, 4,  1,  7,  10, 11, 6,  8,  5,  3,  15, 13, 0,  14, 9],
        [14,    11, 2,  12, 4,  7,  13, 1,  5,  0,  15, 10, 3,  9,  8,  6],
        [4, 2,  1,  11, 10, 13, 7,  8,  15, 9,  12, 5,  6,  3,  0,  14],
        [11,    8,  12, 7,  1,  14, 2,  13, 6,  15, 0,  9,  10, 4,  5,  3]
    ],
    [
        [12,    1,  10, 15, 9,  2,  6,  8,  0,  13, 3,  4,  14, 7,  5,  11],
        [10,    15, 4,  2,  7,  12, 9,  5,  6,  1,  13, 14, 0,  11, 3,  8],
        [9, 14, 15, 5,  2,  8,  12, 3,  7,  0,  4,  10, 1,  13, 11, 6],
        [4, 3,  2,  12, 9,  5,  15, 10, 11, 14, 1,  7,  6,  0,  8,  13]
    ],
    [
        [4, 11, 2,  14, 15, 0,  8,  13, 3,  12, 9,  7,  5,  10, 6,  1],
        [13,    0,  11, 7,  4,  9,  1,  10, 14, 3,  5,  12, 2,  15, 8,  6],
        [1, 4,  11, 13, 12, 3,  7,  14, 10, 15, 6,  8,  0,  5,  9,  2],
        [6, 11, 13, 8,  1,  4,  10, 7,  9,  5,  0,  15, 14, 2,  3,  12]
    ],
    [
        [13,    2,  8,  4,  6,  15, 11, 1,  10, 9,  3,  14, 5,  0,  12, 7],
        [1, 15, 13, 8,  10, 3,  7,  4,  12, 5,  6,  11, 0,  14, 9,  2],
        [7, 11, 4,  1,  9,  12, 14, 2,  0,  6,  10, 13, 15, 3,  5,  8],
        [2, 1,  14, 7,  4,  10, 8,  13, 15, 12, 9,  0,  3,  5,  6,  11]
    ]
]

P = [
    16, 7,  20, 21,
    29, 12, 28, 17,
    1,  15, 23, 26,
    5,  18, 31, 10,
    2,  8,  24, 14,
    32, 27, 3,  9,
    19, 13, 30, 6,
    22, 11, 4,  25
]

P_1 = [
    9,  17, 23, 31,
    13, 28, 2,  18,
    24, 16, 30, 6,
    26, 20, 10, 1,
    8,  14, 25, 3,
    4,  29, 11, 19,
    32, 12, 22, 7,
    5,  27, 15, 21
]


IP = [
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17, 9,  1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7
]
    
IP_1 = [
    40, 8,  48, 16, 56, 24, 64, 32,
    39, 7,  47, 15, 55, 23, 63, 31,
    38, 6,  46, 14, 54, 22, 62, 30,
    37, 5,  45, 13, 53, 21, 61, 29,
    36, 4,  44, 12, 52, 20, 60, 28,
    35, 3,  43, 11, 51, 19, 59, 27,
    34, 2,  42, 10, 50, 18, 58, 26,
    33, 1,  41, 9,  49, 17, 57, 25
]

clair = 0x58A78016670D4CAD
cypher = 0xE9BCADBB461205D5
key = 0x34f17adf58b99264
#0x34f07ade58b89264 may work : to see


faux = [ 0xEBA9ADBB461305D5, 
 0xE9BEADBF461305D5, 
 0xE9BCAFFB461305D5, 
 0xE8ECADBD561305D5, 
 0xE8ECA9BF541205D5, 
 0xE8BCA9BB461005D5, 
 0xE9BCA9BB561207D5, 
 0xE9FCA9BB560605D7, 
 0xE0FCADBA461605D5, 
 0xE9B4ADBA060205D5, 
 0xE9BCA5BB061605D5, 
 0xA9BCBDB2464605D5, 
 0xE9BCBDBB0E4605D5, 
 0xA9BCBDBB465A05D5, 
 0xE9BCBDBB46520DD4, 
 0xA9BCBDBB4212059C, 
 0x89BCADBB421214D4, 
 0xE99CADBB421215D5, 
 0xE9BC8DBB46121495, 
 0xFDBCEC9B46120495, 
 0xF9BCECBB66121495, 
 0xEDBCACBB463205D5, 
 0xE9BCECBB461225D5, 
 0xFDBCECAB461245F5, 
 0x7DBCACAB471205D5, 
 0xE93CADBB471201D5, 
 0xE9BC2DAB461245D5, 
 0xE9B8AD2B471201D1, 
 0xE9B8ADBBC71201D1, 
 0xE9B8ADBB469205C5, 
 0xE9BCADBB461285C1, 
 0xE9B9ADFB46130551 ]


#useful for finding an inverse matrix
def inversePermutationMatrix(M,size):
    new = []
    for i in range(size):
        for j in range(size):
            if M[j] == (i+1):
                new.insert(i, j+1)
           
                
                
    return new

#find common data on a set of list
def intersection(liste):
    result = set(liste[0])
    for l in liste[1:]:
        result.intersection_update(l)
    return result.pop()

#set a Bit in k position for a n bit block gathered
def setBit(n,k): 
    return ((1 << k) | n) 

#finding line and column of a block of 6 bits (for evert sbox)
def computeSbox(block):
     line = (block  & 0b100001)  
     liner=0
     if len(bin(line)[2:]) == 6:     
            if bin(line)[2:3] == "1":
                liner = liner + 2
            if bin(line)[7:8] == "1":
                liner = liner + 1
     elif len(bin(line)[2:]) < 6:
            lenn = bin(line)[2:]
            if lenn[len(lenn)-1:len(lenn)] == "1":
                liner = liner + 1
     column = (block & 0b011110) >> 1
     return liner,column

#compute sbox number(numberSbox) for a gathered block
def computeSbox2(block,numberSbox):
    line,column = computeSbox(block)
    return Sbox[numberSbox-1][line][column]

#compute sbox line and column for a parts of a xor of "block" and "bruteforce"
#useful for K16 computation 
def computeSbox3(numberSbox,block,bruteforce):
                mask = "111111"
                mask = mask[:(numberSbox*6)+6].zfill((numberSbox*6)+6)
                mask = mask.ljust(48,'0')
                imask = int(mask,2)
                    
                testC = block & imask
                #print(bin(testC))
                testC = testC >> (7 - numberSbox) * 6
                testC = testC ^ bruteforce
                l,r =  computeSbox(testC)
                return l,r
            
def cutXinY(Key,X,Y,on):
    part = 0
    Ki = []
    mask = ""
    if X % Y == 0:
        part = X / Y
    else:
        return Ki
    part = int(part)
    for i in range(part):
        mask = mask + "1"
    #print(mask)
    for i in range(Y):
        tmp = mask
        tmp = tmp[:(i*part)+part].zfill((i*part)+part)
        tmp = tmp.ljust(X,'0')
        imask = int(tmp,2)
        #print(hex(imask))
        res = Key & imask
        if(on == "on"):
            res = res >> part
            on = "off"
        Ki.append(res)
    return Ki
        
        

#do a n-shift to the left for a gathered block eg: 10010 -> 00101 for 1-shift
def leftCirculatShift(bit,shift,size):
    return ((bit << shift) % (1 << size)) | (bit >> (size - shift))


#Feistel function aggregates enxpansion and sbox computation and P permutation
def F(RightP,Ki):
   # print("F")
    T = permutation(RightP,E,32)
    T = T ^ Ki
    liner = 0
    res = 0
    tmp = 0
    #print(hex(T))
    cutted = cutXinY(T,48,8,"off")
    for block,i in zip(cutted,range(8)):
        tmp = block >> (42 - (i*6))
        liner,column = computeSbox(tmp)       
        res = res | (Sbox[i][liner][column] << ( 32 - (i+1) * 4 ) ) 
        liner = 0
    return permutation(res,P,32)

#Full computation of DES
def DES(msg,Key):
    Ki = keySchedule(Key)
    Init = permutation(msg,IP,64)
    LR = cutXinY(Init,64,2,"on")  
    for ki in Ki:
        Li1 = LR[1]
        Ri1 = LR[0] ^F(LR[1],ki)      
        LR[0] = Li1
        LR[1] = Ri1      
        res = (LR[1] << 32) | LR[0]
        Res = permutation(res,IP_1,64)
    return Res

#Permute block of size(size) with a table
def permutation(bits,T,size):
    res = 0
    for i in range(len(T)):
       mask = 1 << (size -T[i])
       permut = bool(mask & bits) << (len(T) - i - 1)
       res = res | permut 
    return res


def keySchedule(Key):
    Ki = []
    vi = [1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]  
    T = permutation(Key,PC1,64)
    CD0 = cutXinY(T,56,2,"on")    
    for v in vi:      
      CD0[0] = leftCirculatShift(CD0[0], v, 28)
      CD0[1] = leftCirculatShift(CD0[1], v, 28)
      concC = CD0[0] << 28
      concD = CD0[1]
      conc = concD | concC        
      Ki.append(permutation(conc,PC2,56))      
    return Ki

"""
ATTACK ON DES BY DIFFERENTIAL FAULT ANALYSIS
"""
#find if for each sbox the portion of R15 Xor R15* that are non zeros and put them into candidate for potential key parts for each sbox
def R15XorFaulted():
    sboxAttackContent = {format(i+1): [] for i in range(8)}
    sboxAttackIndex = {format(i+1): [] for i in range(8)}
    
    for i,fault in zip(range(len(faux)),faux):
        R15XoRfaulted = fault ^ cypher
        R15XoRfaulted = permutation(R15XoRfaulted,IP,64)
        
        LR16 = cutXinY(R15XoRfaulted,64,2,"on")
        expansionR16 = bin(permutation(LR16[1],E,32))[2:].zfill(48)
        #print(expansionR16)
        expansionR16_list = [expansionR16[i:i+6] for i in range(0, 48, 6)]
            
        for sboxnumber,block in enumerate(expansionR16_list):
            if block != '000000':
                sboxAttackContent[format(sboxnumber+1)].append(fault)
                sboxAttackIndex[format(sboxnumber+1)].append(i)
    
                
    #print(sboxAttackContent)
    for i in sboxAttackIndex:
        
        print("potential false cypher portion key for sbox " + i)
        print(sboxAttackIndex[i])
    return sboxAttackContent,sboxAttackIndex
                
        #print(expansionR16_list)   


#compute K16 from bruteforcing into the output of the 8 sbox
def crackK16(cypher,faux):
    K16 = 0x000000000000
    sol = {"{}".format(i+1): [] for i in range(8)}
    content,index = R15XorFaulted()
    init = permutation(cypher,IP,64)
    LR16 = cutXinY(init,64,2,"on")
    for i in range(8):
        for j in range(len(content["{}".format(i+1)])):
             
            LR16False = cutXinY(permutation(content["{}".format(i+1)][j],IP,64),64,2,"on")
            
            #P^-1(L16 xor L16*)
            xor = LR16[0] ^ LR16False[0]
            xor = permutation(xor,P_1,32)
            
            #Préparation de E(R15) et E(R15*)
            R15 = permutation(LR16[1],E,32)
            R15false = permutation(LR16False[1],E,32)
            candidate = []
            
            #Finding K16 with brute force costs 2^6 = 64 per j possible false-cypher that went to the i sbox
            #Complexity : 2^9
            for bruteforce in range(64):
                
                #finding line and column for the i^nd sbox for 6 parts bits of R15 ^Bruteforce and R15fals^bruteforce
                line,column = computeSbox3(i,R15,bruteforce)
                fline,fcolumn = computeSbox3(i,R15false,bruteforce)
                
                #S(E(R15) XOR K16) XOR S(E( R15 * ) XOR K16)
                mask = "1111"
                mask = mask[:(i*4)+4].zfill((i*4)+4)
                mask = mask.ljust(32,'0')
                imask = int(mask,2)
                testXor = xor & imask
                testXor = testXor >> (7 - i) * 4
                testSbox = Sbox[i][line][column] ^ Sbox[i][fline][fcolumn]
                
                #test P-1(R15 XOR R15*) = S(E(R15) XOR K16) XOR S(E(R15 *) XOR K16)
                if(testXor == testSbox):
                    candidate.append(bruteforce)
                    
            # adding all viable solution to the list of solution for the i^nd sbox        
            sol["{}".format(i+1)].append(candidate)
        #intersection of the righ bits parts for the i^nd sbox
        solution = intersection(sol["{}".format(i+1)])
        #concatenate the right bits parts
        K16 = K16 << 6
        K16 = K16 | solution 
    
        print("Sbox", i + 1)
        print("Potential solution")
        print(sol["{}".format(i+1)])
        print("Solution", i+1 , "=", hex(solution), "=",solution )
        print("current K16 =", hex(K16))
        print()
            
    print("K16 = ",hex(K16))        
    return K16

#find K using bruteforce on 8 missing bits, then setting paritybit to find the Key
def crackK(cypher,faux):
    K16 = crackK16(cypher,faux)
    PC2K56 = permutation(K16,PC_2,48)
    PC1K48 = permutation(PC2K56,PC_1,56)
    bruteforce = list(product([0,1],repeat=8))
    lostPosition = findPosition(PC_1,PC_2)
    lostP = recoverPosition(lostPosition)
    
    for pos in bruteforce:
        tryb = 0
        for i in range(8):
            if pos[i] == 1:   
                tryb = setBit(tryb, lostP[i])
                
        tryb = tryb | PC1K48
        
        if cypher == DES(clair,tryb):
            return setParityBit(tryb)
 
#set Parity bit as described in DES 
def setParityBit(Key):
    k = bin(Key)[2:]
    k = k.zfill(64)
    count = 0
    final = ""
    for i in range(0,64,8):
        byte = k[i:i+8]
        for j in range(8):
            if k[i+j-1:i+j] == '1':
                count = count + 1
        if count % 2 == 0:
             byte = byte[:-1] + '1' 
        final += byte
        count = 0
    return int(final,2)
    

### Allow to find bits position that are lost after PC operation on reverse keyschedule

def recoverPosition(tab):
    for i in range(8):
        tab[i] = 64 - tab[i]
    print(tab)    
    return tab

def findPosition(PC_1,PC_2):
    tab = []
    ind = []
    for i in range (64):
       if (PC_2[PC_1[i]-1] == 0):
           tab.append(0)
       elif (((PC_1[i]) == 0)):
           tab.append(-1)
       else:
           tab.append(PC_2[PC_1[i]-1])
    print(tab)
    for i,j in zip(tab,range(64)):
        if i == 0:
            ind.append(j+1)
    print(ind)
    return ind


###test phase

K = crackK(cypher,faux)
print("K = ",hex(K))



import PIL
import os
from os import listdir
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import random
from torch.autograd import Variable

#<------------Trainingdatenvorbereiten----------------------->#

#Bilder vereinheitlichen
normalize = transforms.normalize(

    #Durchschnitt Normalisierungsvektor um die Belichtung etc. zu vereinheitlichen
    mean = [0.485, 0.456 , 0.406],

    #Normalverteilung, wie weit diese normal vom Durchschnitt abweichen
    std = [0.229, 0.224, 0.225]
)

transform = transforms.Compose([

    #Größe anpassen
    transforms.Resize(780),
    
    #Breite und Höhe gleiche Größe und zentriert
    transforms.CenterCrop(780),
    
    #als Tensor
    transforms.ToTensor(),
    
    #normalisieren des Bildes
    normalize
])

#leere Listen
train_data_list = []
target_list=[]
train_data=[]

#files enthält direction zum Imageordner
files = listdir('img_train/')

#random Auswahl von Bildern
for i in range(len(listdir('img_train/'))):

    #Random Zahl generieren aus Länge der Filesliste
    f = random.choice(files)

    #Bild f aus files löschen um Mehrfachverwendung auszuschließen
    files.remove(f)

    #Bild laden
    img = PIL.Image.open("img_train/" + f)
    
    #Bild normalisieren - Tensorgröße: (3,780,780) :: 3 weil blau, rot, grün Kanal
    img_tensor = transforms(img)

    #img_tensor.unsqueeze_(0) #macht aus (3, 780, 780) -> (1, 3, 780, 780) - _ bedeutet (inplace), dass auf img_tensor direkt gespeichert wird und nicht nur zurückgegeben wird; 1 bedeutet eine Dimension für die Batchsize
    
    print(img_tensor)

    #Bild der train_data_list Liste anfügen
    train_data_list.append(img_tensor)

    #Image Namen benutzen um Wert zuweisen
    drivingSurfaceDirtRoad = 1 if 'dirt_road_' in f else 0
    drivingSurfaceStoneRoad = 1 if 'stone_road_' in f else 0
    drivingSurfaceStreet = 1 if 'street_' in f else 0

    #OUTPUT: Target:: [DirtRoad, StoneRoad, Street]
    target = [drivingSurfaceDirtRoad, drivingSurfaceStoneRoad, drivingSurfaceStreet]

    #Targetliste wird mit Ergebnis gefüllt
    target_list.append(target)

    #nach 800 Bilder werden die restlichen in einer Batch gespeichert
    if len(train_data_list) >= 64:
        train_data.append((torch.stack(train_data_list)), target_list),
        train_data_list = []
        print('Loaded batch ', len(train_data), 'of ', int(len(listdir('img_train'))))
        print('Percentage Done: ', len(train_data)/int(len(listdir('img_train'))))

#<------------Trainingdatenvorbereiten----------------------->#
#<------------------------------------------------------->#
#<------------Neuronales Netz----------------------->#

class Netz(nn.Module):
    """Das Netz besteht aus 4 Convolutional-Layer, deren Anzahl an Neuronen ansteigt. Es existieren 2 FullyConnected-Layer, welche die Anzahl an Neuronen schrittweise auf 3 verkleinern. 3 deshalb, weil drei mögliche Ergebnisse (Straßentypen) existieren. Das neuronale Netz erkennt, welches der drei Neuronen den höchsten Wert hat und schließt dadurch auf den Straßentyp. Die Forward-Methode bedient sich an diesen Layern und findet von benachbarten Matrixfeldern das Maximum heraus und vereinfacht die Matrix wieder (4x)."""

    #
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, groups=1, bias=True, kernel_size=2, padding=0, stride=1)
        self.conv2 = nn.Conv2d(6, 12, groups=1, bias=True, kernel_size=2, padding=0, stride=1)
        self.conv3 = nn.Conv2d(12, 18, groups=1, bias=True, kernel_size=2, padding=0, stride=1)
        self.conv4 = nn.Conv2d(18, 24, groups=1, bias=True, kernel_size=2, padding=0, stride=1)
        
        #fullyConnectedLayer, der die Martix linearisiert und 3 OUTPUT-Neuronen gibt: Dirt_road;Street; Stone_road
        self.fcl1 = nn.Identity(1000, bias=True)
        self.fcl2 = nn.Identity(3, bias=True)
    
    #
    def forward(self, x):
        
        #x durch erste convolutionalschicht schicken
        x = self.conv1(x)
        
        #das Max der Matrix herausfinden
        x = F.max_pool2d(x, 2, stride=None, padding=0)
        
        #x wieder aktivieren
        x = F.relu(x, inplace=False)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2, stride=None, padding=0)
        x = F.relu(x, inplace=False)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2, stride=None, padding=0)
        x = F.relu(x, inplace=False)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2, stride=None, padding=0)
        x = F.relu(x, inplace=False)

        #erste dimension ignorieren
        x = x.view(-1, 3456)
        x = F.relu(self.fcl1(x))
        x = F.relu(self.fcl2(x))
        x = self.fcl2(x)
        return nn.sigmoid(x)

#Neuronales Netz initialisieren
model = Netz()

#Neuronales Netz in Grafikkarte übergeben
model.cuda()

#<------------Neuronales Netz----------------------->#
#<----------------------------------------------------->#
#<------------Trainingsalgorithmus----------------------->#

#Optimizer ADAM mit Lernrate 0,01 -> optimiert die Parameter des neuronalen Netzes
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

def train(epoch):

    #
    model.train()

    #batch_ID die bis zu einer gewissen Nummer geht. In einer Batch sind 64 Bilder drin -> Zeile 
    batch_ID = 0

    #liste die durch die Trainingsbilder geht
    for data, target in train_data:

        #data wird auf Grafikkarte umgesetzt
        data = data.cuda()

        #target wird als Tensor auf der Grafikkarte ausgeführt
        target = torch.LongTensor(target).cuda()

        #data & target werden als Variablen umgesetzt
        data = Variable(data)
        target = Variable(target)

        #Gradieneten von optimizer auf Null setzen
        optimizer.zero_grad()

        #daten in Model reinstecken
        out = model(data)

        #loss klasssifizieren
        criterion = nn.CrossEntropyLoss()

        #loss berechnen
        loss = criterion(out, target)

        #backpropergaten
        loss.backward()

        #optimizer soll was tun
        optimizer.step()

        #print
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_ID * len(data), len(train_data.dataset),
                .100 * batch_ID / len(train_data), loss.data[0]))

        batch_ID = batch_ID+1

for epoch in range(1, 30):
    train(epoch)

#<------------Trainingsalgorithmus----------------------->#
#<------------------------------------------------------->#
#<------------Testalgorithmus---------------------------->#

def test():
    model.eval()
    files = listdir('img_test/')
    f = random.choice(files)
    img = img.open('img_test/'+ f)
    img_eval_tensor = transforms(img)
    img_eval_tensor.unsqueeze_(0)
    data = Variable(img_eval_tensor.cuda())
    out = model(data)
    print(out.data.max(1, keepdim = True) [1])
    img.show()
    x = input('')

for epoch in range(1,30):
    train(epoch)
    test()

#<------------Testalgorithmus---------------------------->#

#TODO Abspeichern des Machine Learning Models nach jeder Epoche um Trainingsstand abzusichern
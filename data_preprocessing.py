import os
from PIL import Image

def copy(immagini,outputDir,categoryPath):
  #For each image create the path of the mask
  for immagine in immagini:
    #Getting only the name of the image
    nomeImmagine = immagine[:-4]
    percorsoImmagineJpg = os.path.join(categoryPath, nomeImmagine + ".jpg")
    percorsoImmaginePng = os.path.join(categoryPath, nomeImmagine + ".png")
    #Resize at 224 x 224
    imageJpg = Image.open(percorsoImmagineJpg)
    new_imageJpg = imageJpg.resize((224, 224))
    new_imageJpg.save(os.path.join(outputDir, nomeImmagine + ".jpg"))

    imagePng = Image.open(percorsoImmaginePng)
    new_imagePng = imagePng.resize((224, 224))
    new_imagePng.save(os.path.join(outputDir, nomeImmagine + ".png"))

#Downloading and extracting working dataset
%cd datasets
os.environ['KAGGLE_USERNAME'] = "omartornaghi"
os.environ['KAGGLE_KEY'] = "07673df5b4d8e9a4a0983e6eaf21e7aa"
!kaggle datasets download -d alex000kim/magnetic-tile-surface-defects
!unzip magnetic-tile-surface-defects.zip -d MAGNETIC
!rm -rf magnetic-tile-surface-defects.zip
#TRAIN AND TEST SEPARATION
mainPath = '/content/mixed-segdec-net-comind2021/datasets/MAGNETIC'
trainPath = os.path.join(mainPath, 'TRAIN')
testPath = os.path.join(mainPath, 'TEST')
os.mkdir(trainPath)
os.mkdir(testPath)
categorie = os.listdir(mainPath)
for categoria in categorie:
  if('MT_' not in categoria): continue #Skip TRAIN and TEST directories
  categoriaPath = os.path.join(mainPath,categoria, "Imgs")
  immagini = os.listdir(categoriaPath)
  jpg = [k for k in immagini if '.jpg' in k]
  #80% TRAIN 20% TEST
  training = jpg[:int(len(jpg)*0.8)]
  testing = jpg[-int(len(jpg)*0.2):]
  #Copying files
  trainOutputDir = os.path.join(trainPath,categoria,"Imgs")
  testOutputDir = os.path.join(testPath,categoria,"Imgs")
  os.mkdir(os.path.join(trainPath,categoria))
  os.mkdir(os.path.join(testPath,categoria))
  os.mkdir(trainOutputDir)
  os.mkdir(testOutputDir)
  copy(training,trainOutputDir,categoriaPath)
  copy(testing,testOutputDir,categoriaPath)
!rm -rf MAGNETIC/MT_*
%cd ..
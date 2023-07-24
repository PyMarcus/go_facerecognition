package main

import (
	"image"
	"image/color"
	"log"

	"gocv.io/x/gocv"
)

/*

 INSTALATION:

1)go install gocv.io/x/gocv
2)export CGO_CPPFLAGS="-I/usr/local/include"
3)export CGO_LDFLAGS="-L/usr/local/lib -lopencv_core -lopencv_face -lopencv_videoio -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect -lopencv_features2d -lopencv_video -lopencv_dnn -lopencv_xfeatures2d"
4)sudo pacman -S opencv python-opencv

*/
func applyGrayScale(img *gocv.Mat) *gocv.Mat{
	log.Println("Applyng gray scale...")
	grayImg := gocv.NewMat()
	gocv.CvtColor(*img, &grayImg, gocv.ColorBGRToGray)
	return &grayImg
}

func faceRecognition(img *gocv.Mat){
	log.Println("Performing face detection...")

	trainedXMLfile := "./trained_model/frontal.xml"

	classifier := gocv.NewCascadeClassifier()
	defer classifier.Close()

	if !classifier.Load(trainedXMLfile){
		log.Fatalln("Error to load classifier")
	}

	rects := classifier.DetectMultiScale(*img)
	drawRectanglesAroundDetectedFaces(rects, img)

}

func drawRectanglesAroundDetectedFaces(rect []image.Rectangle, img *gocv.Mat){
	for _, r := range rect{
		gocv.Rectangle(img, r, color.RGBA{255, 0, 0, 0}, 2)
	}
}

func showImageDetected(img *gocv.Mat){
	window := gocv.NewWindow("Face detection")
	defer window.Close()
	window.IMShow(*img)

	// wait for a press key
	window.WaitKey(0)
}

func main(){
	log.Println("starting recognition...")
	img := gocv.IMRead("./face.jpg", gocv.IMReadColor)
	if img.Empty() {  
	   log.Fatalln("error to open image")
	}
	defer img.Close()

	log.Println("Image has been found!")

	defer applyGrayScale(&img).Close()

	faceRecognition(&img)

	showImageDetected(&img)

}


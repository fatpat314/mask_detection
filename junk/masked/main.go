package main

import (
    "fmt"

    "github.com/Kagami/go-face"
)

const dataDir = "images"

func main(){
    fmt.Println("Facial racognition")

    rec, err := face.NewRacognizer(dataDir)
    if err != nil{
        fmt.Println("Error, something something")
        fmt.Println(err)
    }
    defer rec.Close()
    )
}

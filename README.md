# Spark-pspectrum

[![Build Status](https://travis-ci.com/sirCamp/spark-pspectrum.svg?branch=master)](https://travis-ci.com/sirCamp/spark-pspectrum)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Scala](https://img.shields.io/badge/scala-v2.11.12-blue)](https://img.shields.io/badge/scala-v2.11.12-blue)

This repository represents a package that contains the PSpectrum computation for character embedding implemented in Spark. 
Furthermore, the repository contains an implementation of a String relaxation in Spark. 
In oder words, this is an useful big data implementation of PSpectrum kernel encoding and String relaxation.



# Usage

## PSpectrum
```scala
var pspectrumEstimator = new PSpectrum()
pspectrumEstimator.setP(3) //setting the degree of the Spectrum
pspectrumEstimator.setInputCol(inputCol) //set input column that must be a string column
pspectrumEstimator.setOutputCol(outputCol) //set output column, it return a VectorUTD encoded as SparseVector of Double

var pspectrumModel = pspectrumEstimator.fit(dataset)

var transformedDataset = pspectrumModel.transform(dataset)

```

## SequenceRelaxation
```scala

val characterEncoding = new mutable.HashMap[String, String]()
val upper_consonant_chars = Array("B", "C", "D", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "X", "Y", "Z")
val lower_consonant_chars = Array("b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "y", "z")
val upper_vowel_chars = Array("A", "E", "I", "O", "U")
val lower_vowel_chars = Array("a", "e", "i", "o", "u")
val number_chars = Array("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")
val symbol_chars = Array("!", "\"", "£", "$", "%", "&", "/", "(", ")", "=", "?", "^", "§", "<", ">", ".", ":", ",", ";", "-", "_", "+", "*", "[", "]", "#")

upper_consonant_chars.foreach(char => {
  characterEncoding.put(char, "C")
})
lower_consonant_chars.foreach(char => {
  characterEncoding.put(char, "c")
})
upper_vowel_chars.foreach(char => {
  characterEncoding.put(char, "V")
})
lower_vowel_chars.foreach(char => {
  characterEncoding.put(char, "v")
})
number_chars.foreach(char => {
  characterEncoding.put(char, "0")
})
symbol_chars.foreach(char => {
  characterEncoding.put(char, "+")
})

var sequenceRelaxation = new SequenceRelaxation()
sequenceRelaxation.setCharacterEncoding(characterEncoding) //setting the degree of the Spectrum
sequenceRelaxation.setInputCol(inputCol) //set input column that must be a string column
sequenceRelaxation.setOutputCol(outputCol) //set output column, it return a VectorUTD encoded as SparseVector of Double


var transformedDataset = sequenceRelaxation.transform(dataset)

```

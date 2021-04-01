# BLL: Funktionale Programmierung für neuronale Netze

This is a project I did for my high school graduation, as such, it is written entirely in German and isn't particularly interesting 
or good. It attempts to introduce a common interface for various neural network architectures in Haskell with a "pure" approach. It 
is mainly based on the backprop Haskell library.

Hier sind die Datein von der BLL. In der Datei Network sind verschiedene 
Netzwerke, Aktivierungsfunktionen, Fehlerfunktionen und viele Hilfsfunktionen.
In Tuple sind Implementierungen von dem Datentyp (:&) und auch inkomplette Instanzen von Matrizen
und Vektoren für die Typklasse Random zur Erzeugung von zufälligen Werten. In Optimierung
sind die Implementierungen von L2, SGD und Minibatch sowie die zwei Typklassen Optimeriung
und seine Alternative Opt. In Zahlen ist ein Programm zur Zahlenerkennung. Zur Verwaltung
von Bibliotheken wurde stack verwendet.

Die Namen einiger Funktionen sind anders as in der BLL (Bsp.: sigmoid statt logisticSchicht).

Teile des Codes, wie die Tuple.hs und Funktionen aus Network.hs sind von Justin Les Repository:
https://github.com/mstksg/inCode, die unter einer [CC-BY-NC-ND 3.0](https://creativecommons.org/licenses/by-nc-nd/3.0/) Lizens stehen.

# pytorch_helper

## Vorgehen für das Erstellen von Pytorch Modellen (universales Vorgehen)

1. **Dataloader** erstellen (Klasse um Daten zu laden für Train/Evaluierung/Testing vom Modell): e.g. (_datareaders.py_)
2. **Architektur (Modell) erstellen** (Layers, Activation-Funktions, Regularization): e.g. (_variational_autoencoder.py_)
3. Modell **trainieren** (_train.py_)
4. Modell **evaluieren** (_evaulation.py_)
5. Modell im **laufenden Betrieb** verwenden

Die Schritte 1-4 werden in diesem Repository als Mini-Projekt erläutert.

### Experiment starten

1. main-Methode von _train.py_ ausführen
2. main-Methode von _evaluation.py_ ausführen

## Wichtige Links

Folgende Links sind sehr hilfreich bei der Entwicklung:


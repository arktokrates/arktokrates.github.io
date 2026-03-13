---
layout: default
permalink: /Testautomatisierung/NeoLoad/
last_modified_at: 2026-03-13
---

# NeoLoad


[Cloud Console](#Cloud-Console)

[Cloud-Lastgenerator](#Cloud-Lastgenerator)

[Lasttest ausführen](#Lasttest-ausführen)

[RealBrowser](#RealBrowser)

[URLs zur Freischaltung](#URLs-zur-Freischaltung)]

&nbsp;

---

&nbsp;


**NeoLoad Quick Links:** https://www.tricentis.com/neoload-quick-links (u.a. Dokumentation, Konto-Zugang und Downloads)




## Cloud Console {#Cloud-Console}

Zugang: [Customer Area](https://www.neotys.com/accountarea/customer-area.html) > [Cloud Console](https://www.neotys.com/accountarea/my-cloud.html)

### IP-Adresse reservieren

- Workgroup auswählen
- Auf **Reserve IP addresses** klicken
- Enddatum des Zeitraums eingeben
- Gewünschte Zone und Anzahl Adressen auswählen

### Cloud-Lastgenerator reservieren

- Load Generators > Reserve a Cloud session
- Workgroup auswählen
- Name der Cloud Session, NeoLoad-Version und E-Mail-Adresse(n) erfassen
- Startdatum wählen; minimale Dauer: 1h
- Gewünschte Stärke der Maschine wählen (Medium, Large, Extra Large)
- Zone und Anzahl der Lastgeneratoren erfassen


### Übersicht über Verbrauch

- Eine Übersicht über die verbrauchten VUH oder Cloud credits unter **Workgroups > My Workgroups**




## Cloud-Lastgenerator {#Cloud-Lastgenerator}

Dieser Cloud-Lastgenerator wird in der Runtime des NeoLoad-Controllers eingebunden über Klick auf + und die Auswahl der zuvor reservierten Cloud-Sitzung. Wenn man die IP des Cloud-Lastgenerators sieht, ist dieser hochgefahren und zur Testdurchführung bereit. Das Hochfahren einer reservierten Maschine dauert etwa fünf Minuten.

Der NeoLoad-Controller prüft, ob die Version etc. stimmt => Farbe des Lastgenerators als rot oder grün dargestellt.

Erst beim Start des Tests werden die **Testobjekte** zum Cloud-Lastgenerator hochgeladen. Die Kommunikation erfolgt über HTTPS.



### Authentifizierung des NeoLoad-Controllers

Der NeoLoad-Controller verbindet sich mit der Cloud-Console über die reservierte IP-Adresse. Der Cloud-Lastgenerator identifiziert den zugehörigen NeoLoad-Controller über einen **Fingerprint**, der vor dem Start des Lasttests in seine Konfiguration geschrieben wird.


Für den Zugriff auf die Lizenz über NeoLoad Web wird der API-Schlüssel benötigt (Verbindung des NeoLoad Controllers mit NeoLoad Web).



## Lasttest ausführen {#Lasttest-ausführen}

- Vor dem Start eines Lasttests die Einstellungen nochmals zu prüfen lohnt sich!


### Populations

- Für jede Population sind die weiteren Einstellungen durchzuführen (Dauer, Lastagenten, Lastverteilung)


### Load Generators

- Wenn man nicht sicher ist, ob ein Lastagent trotz grünem Symbol tatsächlich verfügbar ist, besser kurz auf der jeweiligen Virtual Machine nachprüfen, wie es tatsächlich um dessen Status steht.


### Load Variation Policy

- Auch bei nicht sofortigem Ende des Lasttests, sondern kontinuierlichem Abbau, bleiben auf den Lastgeneratoren oftmals zahlreiche GUI-Fenster offen, so dass aufzuräumen ist.




## RealBrowser {#RealBrowser}

RealBrowser ist eine Technologie zur Aufzeichnung von Skripts in NeoLoad.


### Ausführung über Cloud-Lastgenerator

Die Ausführung eines Testskripts, das mit RealBrowser umgesetzt worden ist, über einen Cloud-Lastgenerator setzt eine Lizenz mit Cloud Credits voraus (eine VUH-Lizenz passt nicht).



## URLs zur Freischaltung {#URLs-zur-Freischaltung}

Diese vier URLs sind gegebenenfalls in der Firewall freizuschalten, um ein Testskript über einen Cloud-Lastgenerator auszuführen:

cloud.saas.neotys.com
neoload-rest.saas.neotys.com
neoload-rest-eu.saas.neotys.com
neoload-api-eu.saas.neotys.com

Sowie zusätzliche die reservierte IP-Adresse des Cloud-Lastgenerators.

In der Firewall darf nur Certificate-Inspection erfolgen (keine Deep Inspection).






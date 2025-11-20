---
layout: default
permalink: /Testautomatisierung/NeoLoad/
last_modified_at: 2025-11-20
---

# NeoLoad


[Cloud Console](#Cloud-Console)

[Cloud-Lastgenerator](#Cloud-Lastgenerator)

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

Der NeoLoad-Controller prüft, ob die Version etc. stimmt => Farbe des Lastgenerators als rot oder grün dargestellt.

Erst beim Start des Tests werden die **Testobjekte** zum Cloud-Lastgenerator hochgeladen.

Wenn man die IP des Cloud-Lastgenerators sieht, ist dieser hochgefahren und zur Testdurchführung bereit.

Der Cloud-Lastgenerator identifiziert den zugehörigen NeoLoad-Controller über einen **Fingerprint**, der vor dem Start des Lasttests in seine Konfiguration geschrieben wird.



## RealBrowser {#RealBrowser}

RealBrowser ist eine Technologie zur Aufzeichnung von Skripts in NeoLoad.


### Ausführung über Cloud-Lastgenerator

Die Ausführung eines Testskripts, das mit RealBrowser umgesetzt worden ist, über einen Cloud-Lastgenerator setzt eine Lizenz mit Cloud Credits voraus (eine VUH-Lizenz passt nicht).



## URLs zur Freischaltung {#URLs-zur-Freischaltung}

Diese vier URLs sind gegebenenfalls in der Firewall freizuschalten, um Testskript über einen Cloud-Lastgenerator auszuführen:

cloud.saas.neotys.com
neoload-rest.saas.neotys.com
neoload-rest-eu.saas.neotys.com
neoload-api-eu.saas.neotys.com

Sowie zusätzliche die reservierte IP-Adresse des Cloud-Lastgenerators.

In der Firewall darf nur Certificate-Inspection erfolgen (keine Deep Inspection).






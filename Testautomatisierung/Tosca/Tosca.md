---
author: Bernhard Fuchs
layout: default
permalink: /Testautomatisierung/Tosca/
last_modified_at: 2025-10-21
---

# Tosca

[Automation Agents](#Automation-Agents)

[Buffer](#Buffer)

[If-Ausdrücke](#If-Ausdrücke)

[Klick-Operation](#Klick)

[Schleifen](#Schleifen)

[Struktur eines Testfalls](#Struktur-eines-Testfalls)

[Verifizierung](#Verifizierung)

[Virtuelle Ordner](#Virtuelle-Ordner)

[Zeilenumbruch](#Zeilenumbruch)

[Ziel eines Testfalls](#Ziel-eines-Testfalls)


&nbsp;

---

&nbsp;


## Automation Agents {#Automation-Agents}

Mehrere Tosca-Prozesse «Tricentis.Automation.Agent.exe» lassen sich auf einmal per Powershell stoppen:

`Stop-Process -Name Tricentis.Automation.Agent`



## Buffer {#Buffer}

### Initialisierung

Wenn möglich sollen Buffer bereits im Abschnitt Vorbedingungen z.B. leer initialisiert oder mit einem Wert gesetzt werden.


### Präfix verwenden

Zur gezielten Identifikation empfiehlt sich die Verwendung von Präfixen für Buffernamen, z.B. nach der Applikation wie «SAP_OrderNo» oder der Verwendung.


### Settings buffern

`{S[…]}`, z.B. `{S[Engine.Current Test User]}`


### Teilzeichenfolgen in Buffer schreiben

`{LEFT[start][end][buffername]}`

`{CALC[LEFT("{B[sample]}"; 3)]}`



## If-Ausdrücke {#If-Ausdrücke}

Am besten werden If-Ausdrücke nur in **Clean-Up-Szenarien** im Abschnitt **Vorbedingung** oder unter **Nachbedingung** verwendet, da jeder Testfall eine definierte Eingabe und ein konkretes Ergebnis haben sollte. Testfälle sind sonst deutlich schwieriger zu lesen und der Wartungsaufwand steigt. Diese Unterscheidung mit bedingten Eingaben und Ergebnissen wird besser über das **TestCaseDesign** und mit zugehörigen Bedingungen und Constraints im blauen Bereich gelöst.



## Klick-Operation {#Klick}

Die Verwendung von {CLICK} für eine Klick-Operation entspricht einer direkten Mausoperation und sollte wenn immer möglich vermieden werden, da sie mehr Zeit zur Ausführung benötigt und instabiler ist. Dagegen wird bei der Verwendung von X direkt das Event des betreffenden Kontrollelements ausgeführt.



## Schleifen {#Schleifen}

Schleifen sollten nur wenn unbedingt notwendig verwendet werden, da die Anzahl Durchläufe vorweg nicht bekannt ist und dies dem Charakter eines Testfalls mit vordefiniertem Ergebnis widerspricht. Für wiederholte Abläufe wird besser **Repetition** eingesetzt.



## Struktur eines Testfalls {#Struktur-eines-Testfalls}

Mit Vorteil werden Testfälle mit einer einheitlichen Struktur umgesetzt, auch wenn manchmal ein Ordner leer bleibt (z.B. Verifizierung).

| Strukturelement                                         | Beschreibung                                                                                                              |
|:---------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|
| Vorbedingungen/Prerequisites/Precondition/Preprocessing | Alles, was vor dem eigentlichen Testprozess auszuführen ist                                                               |
| Prozess/Process/Test content                            | Tatsächlicher Testprozess, der einer Anforderung (Requirement) entsprechen kann                                           |
| Verifizierung/Verification                              | Zu prüfende Elemente                                                                                                      |
| Nachbedingungen/Postcondition/Postprocessing            | Alles, was nach dem Testprozess auszuführen ist, z.B. Löschen generierter Elemente<br>oder Zurücksetzen der Konfiguration |

Die Schritte zur Verifizierung können auch Teil des eigentlichen Testprozesses sein.



## Verifizierung {#Verifizierung}

Jeder Testfall sollte mindestens einen Verifizierungsschritt umfassen und ein klares Ziel haben – ebenso Testfälle zur Erzeugung von Testdaten.



## Virtuelle Ordner {#Virtuelle-Ordner}

Übersicht über alle erforderlichen Objekte bei der Erarbeitung eines Testfalls: virtuellen Ordner auf derselben Ebene wie das TestTemplate anlegen

Query: `=>OBJECTS("<UniqueId>","<UniqueId>" etc.)`

- Module
- TestSheet
- ExecutionList(s)
- Ordner unter Testdaten
- TestEvents
- Andere virtuelle Ordner, z.B. zu relevanten aktiven Konfigurationen zum Hineinziehen
- etc.

==> rascher Zugriff auf diese Objekte (als Alternative zu Tabs in gespeicherter Ansicht)



## Zeilenumbruch in TestStepValue {#Zeilenumbruch}

Zeilenumbruch im Feld TestStepValue: ~



## Ziel eines Testfalls {#Ziel-eines-Testfalls}

Jeder Testfall sollte ein einziges klares Ziel oder einen Verwendungszweck haben. Er sollte so nah wie möglich einem tatsächlichen Business-Szenario entsprechen. Nur erforderliche Elemente sollten verifiziert werden.



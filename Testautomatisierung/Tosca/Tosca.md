---
layout: default
permalink: /Testautomatisierung/Tosca/
last_modified_at: 2025-10-21
---

# Tosca

[Automation Agents](#Automation-Agents)

[Buffer](#Buffer)

[Klick-Operation](#Klick)

[Verifizierung](#Verifizierung)

[Virtuelle Ordner](#Virtuelle-Ordner)

[Zeilenumbruch](#Zeilenumbruch)


&nbsp;

---

&nbsp;


## Automation Agents {#Automation-Agents}

Mehrere Tosca-Prozesse «Tricentis.Automation.Agent.exe» lassen sich auf einmal per Powershell stoppen:

`Stop-Process -Name Tricentis.Automation.Agent`



## Buffer {#Buffer}

### Settings buffern

`{S[…]}`, z.B. `{S[Engine.Current Test User]}`


### Teilzeichenfolgen in Buffer schreiben

`{LEFT[start][end][buffername]}`

`{CALC[LEFT("{B[sample]}"; 3)]}`



## Klick-Operation {#Klick}

Die Verwendung von {CLICK} für eine Klick-Operation entspricht einer direkten Mausoperation und sollte wenn immer möglich vermieden werden, da sie mehr Zeit zur Ausführung benötigt und instabiler ist. Dagegen wird bei der Verwendung von X direkt das Event des betreffenden Kontrollelements ausgeführt.



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


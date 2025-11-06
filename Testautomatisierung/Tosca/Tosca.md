---
author: "Bernhard Fuchs"
layout: default
permalink: /Testautomatisierung/Tosca/
last_modified_at: 2025-11-06
---

# Tosca

[Automation Agents](#Automation-Agents)

[Browser](#Browser)

[Buffer](#Buffer)

[Business Testcase](#Business-Testcase)

[Condition](#Condition)

[Configurations](#Configurations)

[Connection String](#Connection-String)

[Feiertage](#Feiertage)

[Fenster](#Fenster)

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



## Browser {#Browser}

### TestConfigurationParameters

Zur Testausführung über Sauce Labs oder Browsterstack können TestConfigurationParameter verwendet werden:

![Tosca: TestConfigurationParameter für Browser mit Browserstack](/assets/images/Tosca_Browser_TCP_BrowserStack_Proxy_1.png)

![Tosca: TestConfigurationParameter für Browser mit Browserstack](/assets/images/Tosca_Browser_TCP_BrowserStack_Proxy_2.png)

![Tosca: TestConfigurationParameter für Browser mit Sauce Labs](/assets/images/Tosca_Browser_TCP_Saucelabs_Proxy.png)

Vgl. die Dokumentation unter [Remote-Browser steuern](https://docs.tricentis.com/tosca-2025.1/de-de/content/engines_3.0/xbrowser/xbrowser_steer_remote_browser.htm)


### Zurückgehen im Browser mit Javascript

Vgl. https://support.tricentis.com/community/article.do?number=KB0014887

Module «Execute JavaScript» aus Subset Standard.tsu:
- Titel des Tabs (z.B. \*coop.ch)
- JavaScript: `window.history.go(-1)` // auch mehrere Schritte möglich

**Alternative:** Verwendung des Moduls «TBox Sendkeys»



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



## Business Testcase {#Business-Testcase}

Zur Abbildung eines End-to-End-Prozesses (die Business-Testfälle selbst im blauen Bereich lassen sich nicht ausführen – im grünen Bereich: orange Business-ExecutionList)

**Ziel:** einfachere Wartung und besserer Überblick (situationsspezifisch anwenden), unter Verwendung von TDS

* Requirements: Highlevel-Prozesssicht
* Blauer Bereich: Testfälle (einzeln hineingezogen)
* Grüner Bereich: Ausführungsliste (TestExecutionEntry einzeln hineingezogen)

* **Reporting** z.B. bezogen auf Business Testcase



## Condition {#Condition}

Grundsatz: Besser Conditions verwenden als if-Strukturen

Auf z.B. RTSBs lassen sich keine Buffer in Bedingungen verwenden, da diese zum Zeitpunkt der Instanziierung nicht existieren.



## Configurations {#Configurations}

Die im blauen Bereich gesetzten Konfigurationen gelten auch im grünen Bereich, d.h. im grünen Bereich können sie nicht überschrieben werden.



## Connection String {#Connection-String}

Server=toscaprod-db;Database=ToscaProd;Integrated Security=SSPI



## Feiertage {#Feiertage}

Um die Arbeitstage zu berechnen, braucht Tosca Informationen über die Feiertage im jeweiligen Land. Diese lassen sich in den Einstellungen unter **Settings > TBox > Special Dates** definieren (vgl. [«Einstellungen - Special Dates»](https://docs.tricentis.com/tosca-2025.1/de-de/content/tosca_commander/settings_tbox_special_dates.htm?Highlight=special%20dates)).



## Fenster {#Fenster}

Wenn zwei Fenster mit demselben Namen existieren (z.B. in Avaloq): Constraint auf Feld in intendiertem Fenster (z.B. mit `active` oder `exists`) anwenden. So lässt sich das richtige Fenster ansteuern (das Hervorbringen in den Vordergrund ist sonst nicht möglich).



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
| Nachbedingungen/Postcondition/Postprocessing            | Alles, was nach dem Testprozess auszuführen ist, z.B. Löschen generierter Elemente oder Zurücksetzen der Konfiguration |

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





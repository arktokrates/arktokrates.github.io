---
layout: default
permalink: /Konsole/zsh-Shell/
last_modified_at: 2026-03-13
---

# zsh-Shell


[Umgebungsvariablen ändern](#Umgebungsvariablen-ändern)


&nbsp;

---


## Umgebungsvariablen ändern {#Umgebungsvariablen-ändern}

Beispiel: JAVA_HOME

### Prüfen, wo Java installiert ist

`/usr/libexec/java_home -V`

=> `/Library/Java/JavaVirtualMachines/temurin-25.jdk/Contents/Home`


### Shell-Konfiguration mit Editor öffnen

`vi ~/.zshrc`

In Änderungsmodus wechsel: `I`

Zeile ergänzen oder anpassen: `export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-25.jdk/Contents/Home` &nbsp;&nbsp;&nbsp; // Variable global verfügbar machen

Änderungen speichern und Editor schliessen: `Esc` und `:wq`


### Geänderte Shell-Konfiguration dauerhaft speichern

`source ~/.zshrc`


### Wert einer Umgebungsvariable anzeigen

`echo $JAVA_HOME`

=> `/Library/Java/JavaVirtualMachines/temurin-25.jdk/Contents/Home`


---
layout: default
permalink: /Testautomatisierung/RestAssured/
last_modified_at: 2026-03-06
---

# Rest Assured


[Voraussetzungen](#Voraussetzungen)

[Ausführen von Tests](#Ausführen-von-Tests)

[Wichtige Java-Grundlagen](#Wichtige-Java-Grundlagen)

[Kernkonzepte von Rest Assured](#Kernkonzepte-von-Rest-Assured)

[Mini-Übungen (jsonplaceholder)](#Mini-Übungen)

<br>
[Rest Assured mit Kafka](/Testautomatisierung/RestAssured_Kafka)


&nbsp;

---

&nbsp;





## Voraussetzungen {#Voraussetzungen}


### 1. Eine aktuelle Java-Version installieren (hier Temurin):

```bash
brew install --cask temurin
java -version
```

Bei Bedarf zuerst Homebrew installieren: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

Nach der Installation die Java-Version prüfen: `java -version` (allenfalls das Terminal neu starten)



### 2. Maven installieren

```bash
brew install maven
mvn -version
```


### 3. Visual Studio Code für Java einrichten

Diese Extensions installieren und VS Code neu starten:

- Extension Pack for Java (Microsoft)
- Maven for Java (Optional)

&nbsp;

Ein neues Projekt anlegen:

```bash
mkdir restassured-demo
cd restassured-demo
mvn -q archetype:generate \
  -DgroupId=ch.demo \
  -DartifactId=restassured-demo \
  -DarchetypeArtifactId=maven-archetype-quickstart \
  -DarchetypeVersion=1.4 \
  -DinteractiveMode=false
cd restassured-demo
```

&nbsp;

Danach den Inhalt in der Datei `restassured-demo/pom.xml` durch diesen ersetzen:

```xml
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>

  <groupId>ch.demo</groupId>
  <artifactId>restassured-demo</artifactId>
  <version>1.0-SNAPSHOT</version>

  <properties>
    <maven.compiler.release>17</maven.compiler.release>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    <junit.jupiter.version>5.10.2</junit.jupiter.version>
    <restassured.version>5.5.0</restassured.version>
  </properties>

  <dependencies>
    <dependency>
      <groupId>io.rest-assured</groupId>
      <artifactId>rest-assured</artifactId>
      <version>${restassured.version}</version>
      <scope>test</scope>
    </dependency>

    <dependency>
      <groupId>org.junit.jupiter</groupId>
      <artifactId>junit-jupiter</artifactId>
      <version>${junit.jupiter.version}</version>
      <scope>test</scope>
    </dependency>
  </dependencies>

  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-surefire-plugin</artifactId>
        <version>3.3.1</version>
        <configuration>
          <useModulePath>false</useModulePath>
        </configuration>
      </plugin>
    </plugins>
  </build>
</project>
```




## Ausführen von Tests {#Ausführen-von-Tests}

Die folgenden Befehle werden jeweils im Projekt-Hauptverzeichnis (Root) ausgeführt, in dem sich `pom.xml` befindet.

<br>
Alle Tests mit Maven ausführen: `mvn test`

<br>
Nur eine Testklasse ausführen:

`mvn -Dtest=ApiSmokeTest test` oder voll qualifiziert: `mvn -Dtest=ch.demo.ApiSmokeTest test`

<br>
Eine einzelne Test-Methode ausführen:

`mvn -Dtest=ApiSmokeTest#yourTestMethodName test`


<br>
Mehrere Testklassen ausführen:


`mvn -Dtest=ApiSmokeTest,AppTest test`

`mvn -Dtest=Api* test` (mit Wildcard)

Anmerkung: Für Integrationstests über Failsafe:  `-Dit.test=...` (anstelle von `-Dtest=...`)




## Wichtige Java-Grundlagen {#Wichtige-Java-Grundlagen}


### 1. Imports & static imports

Rest Assured nutzt oft static imports, damit `given()`, `when()`, `then() kurz bleiben.

```java
import static io.restassured.RestAssured.*;
import static org.hamcrest.Matchers.*;
```


### 2. JUnit 5

- Ein Test ist einfach eine Methode mit `@Test`.
- Man braucht fast nur:
	- `@Test`
	- optional `@BeforeAll`, `@BeforeEach`

```java
import org.junit.jupiter.api.Test;
```


### 3. Strings und einfache Typen

- IDs sind oft `int/long`.
- JSON-Felder sind meist `String`, `boolean`, `int`.


### 4. Collections

- Häufig braucht man `List<T>` und `Map<String, Object>`:

```java
import java.util.List;
import java.util.Map;
```


### 5. (Optional, aber sehr hilfreich) Java Records als DTO/POJO (ab Java 16+)

Perfekt für JSON-Objekte:

```java
public record Post(int userId, int id, String title, String body) {}
```



## Kernkonzepte von Rest Assured {#Kernkonzepte-von-Rest-Assured}


### 1. given/when/then

- `given()` = Request vorbereiten
- `when()` = Request ausführen
- `then()` = Assertions


### 2. JSONPath Assertions

- Felder prüfst du mit `title` oder `userId` etc.
- Bei Arrays: `size()`, `[0].id` usw.


### 3. Response extrahieren

Für spätere Prüfungen oder Weiterverarbeitung:

- `extract().path("title")`
- oder `extract().as(Post.class)` (Deserialisierung)


### 4. Logging (sehr wichtig beim Lernen)

- `log().all()` zeigt Request/Response.




## Mini-Übungen (jsonplaceholder) {#Mini-Übungen}


### Übung 1: Status und Feld prüfen

```java
import org.junit.jupiter.api.Test;
import static io.restassured.RestAssured.*;
import static org.hamcrest.Matchers.*;

class JsonPlaceholderSmokeTest {

  @Test
  void getPost1_shouldReturn200_andCorrectId() {
    given()
      .baseUri("https://jsonplaceholder.typicode.com")
    .when()
      .get("/posts/1")
    .then()
      .statusCode(200)
      .body("id", equalTo(1))
      //.body("title", not(isEmptyOrNullString()))   // Veraltete Methode
      .body("title", allOf(notNullValue(), not(emptyString())));
  }
}
```


### Übung 2: Query-Parameter und Liste prüfen

```java
class JsonPlaceholderQueryTest {

  @Test
  void getPostsByUser_shouldReturnList() {
    given()
      .baseUri("https://jsonplaceholder.typicode.com")
      .queryParam("userId", 1)
    .when()
      .get("/posts")
    .then()
      .statusCode(200)
      .body("size()", greaterThan(0))
      .body("[0].userId", equalTo(1));
  }
}
```


### Übung 3: Response extrahieren (ohne POJO)

```java
class JsonPlaceholderExtractTest {

  @Test
  void extractTitle() {
    String title =
      given().baseUri("https://jsonplaceholder.typicode.com")
      .when().get("/posts/1")
      .then().statusCode(200)
      .extract().path("title");

    // Beispiel-Check (nur: nicht leer)
    org.junit.jupiter.api.Assertions.assertFalse(title.isBlank());
  }
}
```


### Übung 4: Deserialisieren in ein Record (sehr nah an Rest Assured)

```java
class JsonPlaceholderPojoTest {

  record Post(int userId, int id, String title, String body) {}

  @Test
  void deserializePost() {
    Post post =
      given().baseUri("https://jsonplaceholder.typicode.com")
      .when().get("/posts/1")
      .then().statusCode(200)
      .extract().as(Post.class);

    org.junit.jupiter.api.Assertions.assertEquals(1, post.id());
    org.junit.jupiter.api.Assertions.assertFalse(post.title().isBlank());
  }
}
```


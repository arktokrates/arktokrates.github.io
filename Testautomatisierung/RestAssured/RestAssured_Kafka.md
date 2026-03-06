---
layout: default
permalink: /Testautomatisierung/RestAssured_Kafka/
last_modified_at: 2026-03-06
---

# Rest Assured mit Kafka


[Voraussetzungen](#Voraussetzungen)

[Docker Compose für Kafka](#Docker-Compose-für-Kafka)

[Allgemeiner Ablauf](#Allgemeiner-Ablauf)

Einfacher Test mit Rest Assured](#Einfacher-Test-mit-Rest-Assured)



&nbsp;

---

&nbsp;



## Voraussetzungen

Aktuell existiert meines Wissens keine dauerhaft frei zugängliche Kafka-Instanz, gegen die man Tests schreiben könnte. Daher verwende ich hier ein Docker-Image von `confluentinc/cp-kafka` (Official Confluent Docker Image for Kafka).

[Docker Desktop](https://www.docker.com/products/docker-desktop/) sollte in einer aktuellen Version installiert sein, z.B. für [Mac](https://docs.docker.com/desktop/setup/install/mac-install/).

Mit Rest Assured wird Kafka typischerweise indirekt getestet, über einen REST Proxy, nicht der Broker selbst.



## Docker Compose für Kafka {#Docker-Compose-für-Kafka}

Am gewünschten Ort diese YAML-Datei unter dem Namen `docker-compose.yml` speichern:

```yaml
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.6.1
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"
    healthcheck:
      test: ["CMD-SHELL", "bash -lc 'cat < /dev/null > /dev/tcp/localhost/2181'"]
      interval: 5s
      timeout: 3s
      retries: 30

  kafka:
    image: confluentinc/cp-kafka:7.6.1
    depends_on:
      zookeeper:
        condition: service_healthy
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: "zookeeper:2181"
      KAFKA_LISTENERS: "PLAINTEXT://0.0.0.0:29092,PLAINTEXT_HOST://0.0.0.0:9092"
      KAFKA_ADVERTISED_LISTENERS: "PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092"
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: "PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT"
      KAFKA_INTER_BROKER_LISTENER_NAME: "PLAINTEXT"
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
    healthcheck:
      test: ["CMD-SHELL", "cub kafka-ready -b kafka:29092 1 30"]
      interval: 5s
      timeout: 10s
      retries: 20

  rest-proxy:
    image: confluentinc/cp-kafka-rest:7.6.1
    depends_on:
      kafka:
        condition: service_healthy
    ports:
      - "8082:8082"
    environment:
      KAFKA_REST_HOST_NAME: "rest-proxy"
      KAFKA_REST_LISTENERS: "http://0.0.0.0:8082"
      KAFKA_REST_BOOTSTRAP_SERVERS: "PLAINTEXT://kafka:29092"
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8082/ >/dev/null || exit 1"]
      interval: 5s
      timeout: 5s
      retries: 30
```

&nbsp;

Die drei Services `zookeeper`, `kafka` und `rest-proxy` bauen aufeinander auf. Umgesetzt ist deren Start so, dass zuerst jeweils geprüft wird, ob der vorherige Service läuft, bevor der nächste gestartet wird.

Zuerst die Desktop-Applikation für Docker starten und danach in der Konsole (Bash) diese Docker-Umgebung für Kafka starten:

```bash
docker compose up -d
```

Mit diesem Befehl lässt sich prüfen, ob die drei Services laufen:

```bash
docker compose ps
```

&nbsp;

Der REST Proxy läuft unter: `http://localhost:8082

Prüfen, ob der REST Proxy verfügbar ist:
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8082/
```

<br>
Um die Docker-Services wieder herunterzufahren, dient dieser Befehl:

```bash
docker compose down
```






## Allgemeiner Ablauf {#Allgemeiner-Ablauf}

Um besser zu verstehen, wie ein Test mit Rest Assured gegenüber Kafka aufgebaut werden kann, folgen zunächst die einzelnen Schritte. Diese lassen sich gleich in der Kommandozeile ausführen, so dass das Docker-Image auf korrekte Funktion geprüft werden kann.
Grundsätzlich wird in Kafka ein Event (Topic) produziert, das über eine Subskription konsumiert werden kann.


### 1. Event senden

Hier wird das Topic demo-topic` selbst angelegt.

```bash
curl -s -X POST "http://localhost:8082/topics/demo-topic" \
  -H "Content-Type: application/vnd.kafka.json.v2+json" \
  -d '{"records":[{"value":{"id":123,"msg":"hello"}}]}'
```


### 2. Consumer anlegen

```bash
curl -s -X POST "http://localhost:8082/consumers/demo-consumer-group" \
  -H "Content-Type: application/vnd.kafka.v2+json" \
  -H "Accept: application/vnd.kafka.v2+json" \
  -d '{"name":"c1","format":"json","auto.offset.reset":"earliest"}'
```

Damit ergibt sich eine `base_uri` (z.B. `http://rest-proxy:8082/consumers/demo-consumer-group/instances/c1`).


### 3. Topic abonnieren

```bash
curl -s -X POST "http://localhost:8082/consumers/demo-consumer-group/instances/c1/subscription" \
  -H "Content-Type: application/vnd.kafka.v2+json" \
  -d '{"topics":["demo-topic"]}'
```


### 4. Records lesen

```bash
curl -s -X GET "http://localhost:8082/consumers/demo-consumer-group/instances/c1/records" \
  -H "Accept: application/vnd.kafka.json.v2+json"
```




## Einfacher Test mit Rest Assured {#Einfacher-Test-mit-Rest-Assured}


### 1) Minimale Abhängigkeiten im `pom.xml` definieren

```xml
<dependencies>
  <dependency>
    <groupId>io.rest-assured</groupId>
    <artifactId>rest-assured</artifactId>
    <version>5.5.0</version>
    <scope>test</scope>
  </dependency>
  <dependency>
    <groupId>org.junit.jupiter</groupId>
    <artifactId>junit-jupiter</artifactId>
    <version>5.10.3</version>
    <scope>test</scope>
  </dependency>
</dependencies>
```



### 2) Test schreiben

Testklasse z.B. unter `src/test/java/.../KafkaRestProxyIT.java`

```java
import io.restassured.RestAssured;
import io.restassured.response.Response;
import org.junit.jupiter.api.Test;

import java.time.Duration;

import static io.restassured.RestAssured.given;
import static org.hamcrest.Matchers.*;

class KafkaRestProxyIT {

  private static final String REST_PROXY = "http://localhost:8082";
  private static final String TOPIC = "demo-topic";

  @Test
  void produceAndConsumeJsonRecord() {
    RestAssured.baseURI = REST_PROXY;

    String suffix = String.valueOf(System.currentTimeMillis());
    String group = "demo-consumer-group-" + suffix;
    String instance = "c1-" + suffix;

    // 1) Create consumer instance pro Testlauf
    given()
      .contentType("application/vnd.kafka.v2+json")
      .accept("application/vnd.kafka.v2+json")
      .body("{\"name\":\"" + instance + "\",\"format\":\"json\",\"auto.offset.reset\":\"earliest\"}")
    .when()
      .post("/consumers/" + group)
    .then()
      .statusCode(200);

    // 2) Subscribe
    given()
      .contentType("application/vnd.kafka.v2+json")
      .body("{\"topics\":[\"" + TOPIC + "\"]}")
    .when()
      .post("/consumers/" + group + "/instances/" + instance + "/subscription")
    .then()
      .statusCode(anyOf(is(204), is(200)));

    // 3) Produce
    given()
      .contentType("application/vnd.kafka.json.v2+json")
      .body("{\"records\":[{\"value\":{\"id\":123,\"msg\":\"hello\"}}]}")
    .when()
      .post("/topics/" + TOPIC)
    .then()
      .statusCode(anyOf(is(200), is(202)));

    // 4) Consume durch Polling von Records, bis etwas kommt (maximal 5 s)
    Response r = pollForRecords(group, instance, Duration.ofSeconds(5));

    r.then()
      .statusCode(200)
      .body("$", not(empty()))
      .body("[0].value.id", equalTo(123))
      .body("[0].value.msg", equalTo("hello"));

    // Optional: Consumer sauber schliessen
    given()
      .when()
      .delete("/consumers/" + group + "/instances/" + instance)
      .then()
      .statusCode(anyOf(is(204), is(200), is(404)));
  }

  private static Response pollForRecords(String group, String instance, Duration timeout) {
    long deadline = System.currentTimeMillis() + timeout.toMillis();
    Response last = null;

    while (System.currentTimeMillis() < deadline) {
      last = given()
        .accept("application/vnd.kafka.json.v2+json")
      .when()
        .get("/consumers/" + group + "/instances/" + instance + "/records");

      if (last.statusCode() == 200 && !last.jsonPath().getList("$").isEmpty()) {
        return last;
      }
      try { Thread.sleep(300); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }
    return last;
  }
}
```


In diesem Test wird jeweils eine neue Group und Instance verwendet. Denn wenn die Consumer-Group bereits einmal gelaufen ist, weist sie ein Offset am Ende auf. Dann ist beim nächsten Lauf oft nichts mehr zu lesen.
Und da `/records` leer sein kann, wenden wir ein Polling auf den Record an.




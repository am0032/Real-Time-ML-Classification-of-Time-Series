void setup() {
  // Initialize serial communication at a baud rate of 9600
  Serial.begin(9600);
}

void loop() {
  // Read the analog value from a sensor (e.g., connected to A0)
  int sensorValue = analogRead(A0);

  // Send the sensor value as a string over serial
  Serial.println(sensorValue);

}

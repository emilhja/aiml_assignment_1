angående C.py så ska jag lösa detta. har du några tips på vart jag kan läsa för att förstå detta? ## (C) ✅ ANN-lager:
  PyTorch version:

  - Använd PyTorch 2.1 (eller bättre). Använd helst Python 3.10 (eller bättre).
  - Kopplas först ihop alla lager i perceptronen så att du får en PyTorch-modell (a.k.a. module). Denna definierar i detalj
  compute-grafen för din perceptron.
  - Använd därefter din perceptron via PyTorch. Googla själv för att få information om hur detta går till rent praktiskt. Det
  finns gott om information på webben kring PyTorch!
  - I denna version ska även träning av nätverket ske, d.v.s. vi ska loopa över epochs, och applicera back-prop. En
  vidareutveckling av back-prop som kallas ADAM brukar användas eftersom den är både snabb och inte lika ofta fastnar i
  dåliga lokala minima, jämfört med ren back-prop.
  - Se avsnittet “Tips för (C)” nedan.


• Jag kollar den officiella PyTorch-dokumentationen och plockar ut de delar som bäst matchar just del (C): nn.Module,
  träningsloop, autograd och Adam.

• Searching the web

• Searched site:pytorch.org PyTorch beginner tutorial nn.Module autograd optimizer Adam

• Searching the web

• Searched site:pytorch.org PyTorch nn.Module tutorial beginner basics optimization Adam

──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• Ja. För just del (C) skulle jag läsa i den här ordningen, och hålla mig till PyTorchs egna tutorials:

  1. nn.Module och hur man bygger modellen:
     https://docs.pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
  2. Enkel modellstruktur med nn.Linear, nn.ReLU, nn.Sequential:
     https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
  3. autograd och varför loss.backward() fungerar:
     https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html
  4. Träningsloop med optimizer.zero_grad(), loss.backward(), optimizer.step():
     https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
  5. Hela nybörjarspåret om du vill se helheten:
     https://docs.pytorch.org/tutorials/beginner/basics/intro.html

  Så skulle jag översätta uppgiften till PyTorch-begrepp:

  - “koppla ihop alla lager” = skapa en klass som ärver från torch.nn.Module
  - “compute-grafen” = det som byggs när du kör forward(...)
  - “träning” = loop över epochs där du räknar loss och kör backprop
  - “ADAM” = torch.optim.Adam(model.parameters(), lr=...)

  Minimal mental modell för (C):

  model = MyNetwork()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  loss_fn = torch.nn.MSELoss()   # eller annan beroende på uppgiften

  for epoch in range(num_epochs):
      y_pred = model(x_train)
      loss = loss_fn(y_pred, y_train)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  Om du vill förstå detta snabbt, fokusera på fyra saker först:

  - nn.Module
  - forward
  - loss.backward()
  - torch.optim.Adam

  Om du vill kan jag också öppna C.py och säga exakt vilka delar du behöver förstå först, utifrån just din kod.
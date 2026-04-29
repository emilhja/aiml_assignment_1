# Assignment-1-part-2: Power-ups

![Screenshot 2025-04-08 at 14.01.14.png](attachment:bf9cef5c-9d0a-4b92-922a-2ec51fa4c21d:Screenshot_2025-04-08_at_14.01.14.png)

Eftersom jag vill undvika att vara för mycket “curling-lärare”, så är även delmoment 2 utformat så att ni ska behöva leta rätt på information själva kring vilka metoder som finns för att förbättra perceptronens prestation, learning, och motståndskraft mot overfitting. Både googling och att fråga Chat är utmärkta metoder.

<aside>
💡

Det är bättre att ni blir vana vid att hitta information själva, än att jag paketerar det i färdigtuggat format - men ni är mycket välkomna att fråga om informationen ni hittar online känns otydlig eller obegriplig! Fastna inte - då är det bättre att fråga!

</aside>

## Syfte med del 2

Del 1 handlade om att nå “Hello World”-nivån för machine learning. Del 2 är lite mer avancerad och syftar till att ge er erfarenhet av:

1. hur man kan jobba med PyTorch för att få fram bästa möjliga modell, lösa problem, undvika overfitting, använda insamlad data effektivt, hyper-parameter tuning, etc
2. att mixa olika typer av layers i era modeller, exempelvis Convolutional layers
3. att förstå poängen med MLOps

## Kravspecifikation för del 2

1. ✅ Börja använda basala **MLOps best-practices:**
    1. 📊 Ordna så att du har ett bra sätt att övervaka träningen, samt att spara (och hålla ordning på!) versioner av modellerna du tränar upp. (Tips: Foldrar, smart valda filnamn, etc).
    2. 💾 Kom ihåg att det troligen inte är den sista epokens checkpoint (sparad version) som är den bästa, p.g.a. overfitting om du kör många epochs. Du behöver själv se till att spara checkpoints lite då och då under träningen.
    3. 🗄️ Versionshantering av parametrar, kod, och checkpoints (modellens parametrar) blir snabbt väldigt viktigt när man börjar jobba seriöst med machine learning. Om man inte håller reda på vilken körning som gjorts med vilka parametrar, så blir det svårt att jämföra prestanda, vidareutveckla modellen senare, etc. Ingen vill ha en modell levererad där programmeraren inte minns hur den gjordes.
    4. 🚀 **Fördjupning/överkurs:** Vilka verktyg finns som kan hjälpa till att hålla ordning på versioner, körningar, resultat, hyper parameters, etc?
2. 🚀 **Fördjupning/överkurs:** Addera **performance metrics**, så att du kan se hur lång tid respektive tränings-körning tar. (En körning = Alla epochs av träning för en given modell med specificerad uppsättning hyper parameters).
3. ✅ Applicera lämpliga **data augmentation** methods för att artificiellt variera, och till och med “förstora” datamängden lite. Använd exempelvis skalning, rotation, färgvariation, brus, (spegelvändning?), etc. Här är lite repetition om data augmentation:
    1. Se avsnittet “[Shared - Uppgift 1 Perceptron för OCR](https://www.notion.so/Shared-Uppgift-1-Perceptron-fo-r-OCR-2b9a7700ac26801d9a3ad9eac0e87b82?pvs=21)” längre ner i denna uppgift (del 3).
    2. Bra 10-minuters video om ämnet: [Pytorch-Data-Augmentation-using-Torchvision](https://youtu.be/Zvd276j9sZ8?si=9fGaNnoRzsHrvxB6)
    3. Observera att det inte är säkert att data augmentation leder till bättre resultat i alla lägen, eftersom det beror på detaljerna i vad man bygger, och vilken data man har.
4. ✅ Det är nu dags att testa CNN istället för FFN. Byt ut de första lagren i modellen till **convolutional layers** för att ta upp translationsinvarians i bilden. Sök själv upp vad man brukar ha ***direkt efter varje convolutional layer*** för att hålla ordning på antalet dimensioner till nästa lager .
    
    > (Svårt ord “translations-invarians”: translation=förflyttning, invarians=”samma oavsett” ⇒ translationsinvarians = ”spelar ingen roll var i bilden”)
    > 
5. ✅ Experimentera därefter med att addera ett till (eller flera) convolutional layer(s). Tanken är att testa **olika nät-arkitekturer**. Använd Google/Chat för initial gissning för modell-arkitektur, men sedan är det trial-and-error som gäller. Att fundera på:
    1. ⚖️ Hur påverkas resultatet av olika modell-arkitektur? Var noga med att inte jämföra modeller med stor skillnad i antal parameters, om aspekter som tex lagertyp ska utvärderas.
    2. 🤔 Vilken typ av features detekteras typiskt av de senare convolutional-lagren jämfört med det första convolutional-lagret?
        1. Hint: Videon från lektion om convolutional layers: [Convolutional-Neural-Networks-Explained](https://youtu.be/pj9-rr1wDhM?si=cjWmR5ets048WjfZ). (Självfallet kan du också plotta weight-matrices från din egen modell också om du vill!)
6. ✅ Applicera lämpliga **regularization methods** för att minimera risken för overfitting och göra cost-function-landskapet fördelaktigt för back-prop.
    1. Exempel på sådana metoder är: drop-out, weight-decay, noise injection, batch normalization, etc.
    2. Mer info om regularization: [understanding-regularization-with-pytorch](https://medium.com/analytics-vidhya/understanding-regularization-with-pytorch-26a838d94058)
7. 🚀 **Fördjupning:** Skapa en lista av kombinationer av hyper parameters, och låt datorn träna din modell på nytt för samtliga hyperparameter settings i din lista. Detta kallas för **hyper-parameter tuning**, d.v.s. att via trial-and-error hitta inställningar som funkar.
    - Enklast möjliga hyper-param-tuning: Gör en lista med t.ex. 10 rader (en per körning), där varje rad specificerar alla hyper-parameters som kan vara svårt att gissa optimalt värde på:
        - Antal neuroner för respektive lager
        - Typ av activation function
        - Vilka lager som ska vara convolutional, storlek på convolutional kernel, antal convolutional kernels per lager, etc.
        - Learning rate / hyper-params till optimizer ADAM
        - Drop-out level
        - Settings för olika typer av data augmentation (rotation, skalning, noise, etc)
    - Det finns även en mängd automatiserade typer av hyper-param-tuning. Se exempelvis: https://en.wikipedia.org/wiki/Hyperparameter_optimization
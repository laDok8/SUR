https://www.fit.vutbr.cz/study/courses/SUR/public/projekt_2022-2023/SUR_projekt2022-2023.zip

Tento archív obsahuje dva adresáře:

train
dev

a každý z těchto adresářů obsahuje podadresáře jejímiž názvy jsou čísla
od 1 do 31, která odpovídají jednotlivým třídám - osobám k identifikaci.
Každý podadresář obsahuje trénovací vzory pro danou třídu ve formátu PNG
a WAV.

Rozdělení dat do adresářů train a dev je možné použít pro trénování a
vyhodnocování úspěšnosti vyvíjeného rozpoznávače, toto rozdělení však není
závazné (např. pomocí technik jako je jackknifing lze efektivně trénovat
i testovat na všech datech). Při pokusech o jiné rozdělení dat může být
užitečné respektovat informace o tom, které trénovací vzory byly pořízený
v rámci jednoho nahrávacího sezení. Jméno každého souboru je rozděleno do
poli pomocí podtržítek (např. f401_01_f18_i0_0.png), kde první pole (f401)
je identifikátor osoby a druhé pole je číslo nahrávacího sezení (01).

Ke trénování rozpoznávačů můžete použít pouze tyto dodané trénovací data.
NENÍ POVOLENO jakékoli využití jiných externích řečových či obrázkových
dat, jakožto i použití již předtrénovaných modelů (např. pro extrakci
reprezentací (embeddings) obličejů nebo hlasu). 

Ostrá data, na kterých budou vaše systémy vyhodnoceny, budou k dispozici
v pátek, 28. dubna ráno. Tato data budu obsahovat řádově stovky souboru
ke zpracování. Vašim úkolem bude automaticky rozpoznat identity osob
v těchto souborech vašimi systémy (věřím Vám že nebudete podvádět a dívat se
na obrázky čí poslouchat nahrávky) a uploadovat soubory s výsledky do IS. 
Soubor s výsledky bude ASCII soubor s 33-mi poli na řádku oddělenými mezerou.
Tyto pole budou obsahovat popořadě následující údaje:

 - jméno segmentu (jméno souboru bez přípony .wav či .png)
 - tvrdé rozhodnutí o třídě, kterým bude celé číslo s hodnotou od 1 do 31.
 - následujících 31 polí bude popořadě obsahovat číselná skóre odpovídající
   logaritmickým pravděpodobnostem jednotlivých tříd 1 až 31. 
   (Pokud použijete klasifikátor jehož výstup se nedá interpretovat
   pravděpodobnostně, nastavte tato pole na hodnotu NaN.

V jakém programovacím jazyce budete implementovat váš rozpoznávač či pomocí
jakých nástrojů (spousta jich je volně k dispozici na Internetu) budete data
zpracovávat záleží jen na Vás. Odevzdat můžete několik souborů s výsledky
(např. pro systémy rozhodujícím se pouze na základě řečové nahrávky či pouze
obrázku). Maximálně však bude zpracováno 5 takových souborů. Každá skupina
musí odevzdat alespoň jeden systém (a výsledky) pro obrázky a jeden pro
nahrávky. Případně můžete odevzdat systém kombinující obě modality.

Soubory s výsledky můžete do soboty 29. dubna 23:59 uploadovat do IS. Klíč
se správnými odpověďmi bude zveřejněn 30. dubna. Své systémy potom budete
prezentovat v krátké prezentaci (5-10min) 3. května na přednášce, kde vám
budou sděleny výsledky.

Na tomto projektu budete pracovat ve skupinách (1-3 lidí), do kterých
se můžete přihlásit ve IS. Jména souborů s výsledky pro jednotlivé
systémy volte tak, aby se podle nich dalo poznat o jaký systém
se jedná (např. audio_GMM, image_linear). Každá skupina uploadne
všechny soubory s výsledky zabalené do jednoho ZIP archívu se jménem
login1_login2_login3.zip či login1.zip, podle toho, kolik
Vás bude ve skupině. Kromě souborů s výsledky bude archív obsahovat
také adresář SRC/, do kterého uložíte soubory se zdrojovými kódy
implementovaných systémů. Dále bude archív obsahovat soubor dokumentace.pdf,
který bude v českém, slovenském nebo anglickém jazyce popisovat Vaše řešení
a umožní reprodukci Vaší práce. Důraz věnujte tomu, jak jste systémy během
jejich vývoje vyhodnocovali, a které techniky či rozhodnutí se pozitivně
projevily na úspěšnosti systému. Tento dokument bude také popisovat jak
získat Vaše výsledky pomocí přiloženého kódu. Bude tedy uvedeno jak Vaše
zdrojové kódy zkompilovat, jak vaše systémy spustit, kde hledat
výsledné soubory, jaké případné externí nástroje je nutné instalovat a
jak je přesně použít, atd. Očekávaný rozsah tohoto dokumentu jsou
3 strany A4. Do ZIP archívu prosím nepřikládejte evaluační data!

Inspiraci pro vaše systémy můžete hledat v archívu demonstračních příkladů
pro předmět SUR:

https://www.fit.vutbr.cz/study/courses/SUR/public/prednasky/demos/

Zvláště se podívejte na příklad detekce pohlaví z řeči: demo_genderID.py
Užitečné vám mohou být funkce pro načítaní PNG souborů (png2fea) a extrakci
MFCC příznaků z WAV souborů (wav16khz2mfcc).

Hodnocení:
- vše je odevzdáno a nějakým způsobem pracuje:
  - čtou se soubory
  - produkuje se skóre
  - jsou správně implementovány a natrénovány nějaké "rozumné" rozpoznávače
    pro obrázky a pro nahrávky a/nebo kombinaci obou modalit (rozpoznávače
    nemusí pracovat se 100% úspěšností, jsou to reálná data!)
  - jsou odevzdány všechny požadované soubory v požadovaných formátech.
  - v dokumentaci vysvětlíte, co, jak a proč jste dělali a co by se ještě dalo zlepšit.
  ... plný počet 25 bodů.

- něco z výše uvedeného není splněno => méně bodů.

Poslední modifikace: 5. dubna 2023, Lukáš Burget
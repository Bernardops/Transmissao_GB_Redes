# Transmissão Digital em Python – Projeto de Redes 2
Feito por: Bernardo Salvador
## 1. Descrição Geral

Este projeto foi desenvolvido como parte do trabalho da disciplina de Redes de Computadores, com foco na aplicação prática dos conceitos de codificação de canal e modulação digital. O objetivo central é analisar como esses elementos influenciam a taxa de erro de bits (BER) e a eficiência espectral de um sistema de comunicação.

Este projeto contém o arquivo com o código, imagens dos grafos, um vídeo demonstrativo e o readme com a descrição do trabalho.

Todo o sistema foi implementado em Python, utilizando apenas NumPy e Matplotlib. A simulação inclui transmissor, codificação de linha, modulação, canal ruidoso, demodulação, cálculo da BER e análise gráfica.

---

## 2. Objetivo do Trabalho

Conforme definido no enunciado do GB, o trabalho envolve:

1. Gerar uma mensagem em ASCII e convertê-la em uma sequência de bits.
2. Aplicar pelo menos uma técnica de codificação de canal (codificação de linha).
3. Implementar duas técnicas de modulação digital, como BPSK e QPSK.
4. Modelar o canal introduzindo ruído (AWGN e ruído em rajada).
5. Realizar a demodulação do sinal recebido.
6. Recuperar os dados transmitidos e comparar com os originais.
7. Calcular a taxa de erro de bits (BER).
8. Avaliar o impacto do ruído e da modulação no desempenho do sistema.
9. Apresentar gráficos relacionando BER e SNR (Eb/N0).
10. Comparar a eficiência das modulações implementadas.

O objetivo final é fornecer uma visão prática e completa do funcionamento de um enlace digital.

---

## 3. Conceitos Utilizados no Projeto

### 3.1 Conversão de Texto para Bits

A mensagem de entrada é convertida para ASCII de 8 bits por caractere. Esse formato facilita a transmissão e mantém a relação direta entre texto e dados binários.

### 3.2 Codificação de Linha

Três técnicas de codificação de canal foram implementadas para demonstração:

- NRZ Polar: bit 1 mapeado em +1 e bit 0 em −1.
- AMI: bits 1 alternam entre +1 e −1; bits 0 são transmitidos como zero.
- Manchester (IEEE 802.3): cada bit possui uma transição de nível no meio do período.

Esses esquemas são apresentados graficamente com o objetivo de ilustrar diferentes formas de representar sinais digitais no meio físico. É possível dar zoom e salvar os plots.

### 3.3 Modulação Digital

Foram aplicadas duas modulações:

- BPSK: cada bit é representado por uma fase diferente (−1 ou +1).
- QPSK (com mapeamento Gray): utiliza dois bits por símbolo, aumentando a eficiência espectral.

Ambas são relevantes em sistemas digitais reais, e acho essas modulações interessantes.

### 3.4 Canal Ruidoso

O sistema permite dois modelos de ruído:

- AWGN: ruído branco gaussiano aditivo, utilizado como referência teórica em comunicações.
- Ruído em rajada (burst noise): gera picos esporádicos de ruído mais intenso, simulando interferências reais.

### 3.5 Eb/N0 (SNR por bit)

A análise do desempenho das modulações é baseada na razão Eb/N0, que expressa a energia por bit em relação ao nível de ruído. 
### 3.6 BER (Bit Error Rate)

A BER é calculada comparando-se os bits transmitidos com os recebidos após demodulação. O procedimento é repetido para cada valor de Eb/N0, permitindo construir curvas de desempenho.

---

## 4. Como Executar o Projeto

### 4.1 Requisitos

- Python 3 Mais recente, e com essas bibliotecas:
- NumPy  
- Matplotlib  

Instalação e execução: Tenha Python instalado no sistema, e rode o arquivo em uma IDE ou digite no terminal de comandos:
pip install numpy matplotlib, seguido de python projeto_transmissao.py

O sistema exibirá menus interativos que permitem, em sequência:

1. Selecionar ou inserir uma mensagem.
2. Visualizar as codificações de linha.
3. Escolher o tipo de canal (AWGN, rajada ou ambos) após fechar a visualização das codificações de linha.
4. Transmitir e reconstruir a mensagem usando BPSK e QPSK.
5. Gerar os gráficos de BER × Eb/N0 (SNR).

---


import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1) Texto <-> bits
# ==========================================================

def texto_para_bits(texto):
    dados = texto.encode("latin1")  # ASCII 8 bits
    bits = []
    for byte in dados:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return np.array(bits, dtype=int)

def bits_para_texto(bits):
    bits = np.array(bits, dtype=int)
    assert len(bits) % 8 == 0
    bytes_list = []
    for i in range(0, len(bits), 8):
        valor = 0
        for b in bits[i:i+8]:
            valor = (valor << 1) | int(b)
        bytes_list.append(valor)
    return bytes(bytes_list).decode("latin1")

# ==========================================================
# 2) Codificação de linha
# ==========================================================

def codificar_ami(bits):
    """
    Codificação bipolar AMI:
    - bit '0' -> nível 0
    - bit '1' -> alterna entre +1 e -1
    """
    niveis = np.zeros(len(bits))
    ultimo_sinal = -1
    for i, b in enumerate(bits):
        if b == 1:
            # alterna entre +1 e -1
            ultimo_sinal *= -1
            niveis[i] = ultimo_sinal
        else:
            niveis[i] = 0
    return niveis

def codificar_manchester(bits):
    """
    Manchester padrão IEEE 802.3:
      - bit '0': alta -> baixa  ( +1 depois -1 )
      - bit '1': baixa -> alta  ( -1 depois +1 )
    """
    sinal = []
    for b in bits:
        if b == 0:
            # 0 = high-to-low
            sinal.extend([1, -1])
        else:
            # 1 = low-to-high
            sinal.extend([-1, 1])
    return np.array(sinal, dtype=float)


def plota_codificacao_linha(bits):
    """
    Plota forma de onda para:
      - NRZ polar
      - AMI
      - Manchester (IEEE 802.3)
    usando TODOS os bits da mensagem.
    """
    if len(bits) == 0:
        print("Nenhum bit para plotar.")
        return

    bits_vis = bits
    N = len(bits_vis)

    # NRZ polar: 0 -> -1, 1 -> +1
    nrz = 2*bits_vis - 1

    ami = codificar_ami(bits_vis)
    man = codificar_manchester(bits_vis)

    # Eixos de tempo (amostra por bit)
    t_nrz = np.arange(N+1)
    t_ami = np.arange(N+1)
    t_man = np.linspace(0, N, 2*N+1)

    plt.figure(figsize=(14, 8))

    # NRZ
    plt.subplot(3, 1, 1)
    plt.step(t_nrz, np.r_[nrz[0], nrz], where='pre')
    plt.ylim(-1.5, 1.5)
    plt.yticks([-1, 0, 1])
    plt.title("Codificação de linha: NRZ polar (todos os bits)")
    plt.grid(True)

    # AMI
    plt.subplot(3, 1, 2)
    plt.step(t_ami, np.r_[ami[0], ami], where='pre')
    plt.ylim(-1.5, 1.5)
    plt.yticks([-1, 0, 1])
    plt.title("Codificação de linha: AMI bipolar (todos os bits)")
    plt.grid(True)

    # Manchester
    plt.subplot(3, 1, 3)
    plt.step(t_man, np.r_[man[0], man], where='pre')
    plt.ylim(-1.5, 1.5)
    plt.yticks([-1, 0, 1])
    plt.title("Codificação de linha: Manchester (IEEE 802.3) (todos os bits)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ==========================================================
# 3) Modulações: BPSK e QPSK (Gray)
# ==========================================================

def bpsk_mod(bits):
    """
    BPSK:
      0 -> -1
      1 -> +1
    """
    return 2*bits - 1  # array de +1/-1

def bpsk_demod(recebido):
    """
    Demodulação BPSK:
    Decisão por limiar em 0.
    """
    bits_hat = np.where(recebido >= 0, 1, 0)
    return bits_hat.astype(int)

def qpsk_mod(bits):
    """
    QPSK com mapeamento Gray:
      b0 b1 -> símbolo (I + jQ)
      0  0  ->  +1 + j*1
      0  1  ->  +1 - j*1
      1  1  ->  -1 - j*1
      1  0  ->  -1 + j*1

    Implementação:
      I = 1 - 2*b0
      Q = 1 - 2*b1

    Se número de bits for ímpar, faz padding com 0 no final.
    """
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)  # padding

    b0 = bits[0::2]
    b1 = bits[1::2]

    I = 1 - 2*b0
    Q = 1 - 2*b1

    simbolos = I + 1j*Q
    return simbolos, len(bits)  # retorna também o tamanho total (com padding)

def qpsk_demod(recebido, n_bits_total):
    """
    Demodulação QPSK coerente:
    Decide o bit pelo sinal da parte real (I) e imaginária (Q).
    """
    I = np.real(recebido)
    Q = np.imag(recebido)

    b0_hat = np.where(I >= 0, 0, 1)
    b1_hat = np.where(Q >= 0, 0, 1)

    bits_hat = np.empty(2*len(b0_hat), dtype=int)
    bits_hat[0::2] = b0_hat
    bits_hat[1::2] = b1_hat

    # Remove eventual bit de padding
    bits_hat = bits_hat[:n_bits_total]
    return bits_hat


# ==========================================================
# 4) Canal (AWGN ou em rajada) e simulação de BER
# ==========================================================

def canal_ruidoso(signal, EbN0_dB, bits_por_simbolo=1,
                  tipo_ruido="awgn", burst_prob=0.01, burst_factor=10):
    """
    Canal genérico:
      - tipo_ruido = "awgn"  -> canal AWGN clássico
      - tipo_ruido = "rajada" -> AWGN com ruído em rajada (burst noise)

    Parâmetros:
      signal           : vetor de símbolos (real ou complexo)
      EbN0_dB          : razão Eb/N0 em dB
      bits_por_simbolo : nº de bits mapeados em cada símbolo (1 para BPSK, 2 para QPSK, etc.)
      burst_prob       : probabilidade de uma amostra cair em uma rajada (apenas se tipo_ruido="rajada")
      burst_factor     : fator de aumento do ruído nas amostras em rajada
    """
    # Converte Eb/N0 de dB para linear
    EbN0_lin = 10**(EbN0_dB / 10.0)

    # Energia média por símbolo
    Es = np.mean(np.abs(signal)**2)
    # Energia média por bit
    Eb = Es / bits_por_simbolo
    # Densidade espectral de ruído
    N0 = Eb / EbN0_lin

    # Variância por dimensão real: N0/2
    sigma2 = N0 / 2.0
    sigma = np.sqrt(sigma2)

    # Gera ruído AWGN base (real ou complexo)
    if np.iscomplexobj(signal):
        ruido = sigma * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    else:
        ruido = sigma * np.random.randn(*signal.shape)

    # Se for ruído em rajada, amplifica algumas amostras
    if tipo_ruido == "rajada":
        # Máscara booleana onde haverá rajada
        mask = np.random.rand(signal.shape[0]) < burst_prob
        escala = np.ones(signal.shape[0])
        escala[mask] = burst_factor

        # Aplica escala ao ruído
        if np.iscomplexobj(ruido):
            ruido = ruido * escala
        else:
            ruido = ruido * escala

    # Retorna sinal recebido
    return signal + ruido


def awgn(signal, EbN0_dB, bits_por_simbolo=1):
    """
    Mantida para compatibilidade:
    Canal AWGN clássico chamando canal_ruidoso com tipo_ruido="awgn".
    """
    return canal_ruidoso(signal, EbN0_dB, bits_por_simbolo=bits_por_simbolo,
                         tipo_ruido="awgn")


def simula_ber_bpsk(bits, EbN0_dBs,
                    tipo_ruido="awgn", burst_prob=0.01, burst_factor=10):
    """
    Simula BER para BPSK em função de Eb/N0 (em dB),
    podendo escolher entre canal AWGN puro ou com ruído em rajada.
    """
    ber = []
    bits = np.array(bits, dtype=int)

    for snr in EbN0_dBs:
        s = bpsk_mod(bits)
        r = canal_ruidoso(s, snr, bits_por_simbolo=1,
                          tipo_ruido=tipo_ruido,
                          burst_prob=burst_prob,
                          burst_factor=burst_factor)
        bits_hat = bpsk_demod(r)
        erros = np.sum(bits != bits_hat)
        ber.append(erros / len(bits))

    return np.array(ber)


def simula_ber_qpsk(bits, EbN0_dBs,
                    tipo_ruido="awgn", burst_prob=0.01, burst_factor=10):
    """
    Simula BER para QPSK em função de Eb/N0 (em dB),
    podendo escolher entre canal AWGN puro ou com ruído em rajada.
    """
    ber = []
    bits = np.array(bits, dtype=int)

    for snr in EbN0_dBs:
        s, n_bits_total = qpsk_mod(bits)
        r = canal_ruidoso(s, snr, bits_por_simbolo=2,
                          tipo_ruido=tipo_ruido,
                          burst_prob=burst_prob,
                          burst_factor=burst_factor)
        bits_hat = qpsk_demod(r, n_bits_total)

        # bits originais com padding (se houver) para comparar
        if len(bits_hat) > len(bits):
            bits_ref = np.append(bits, np.zeros(len(bits_hat) - len(bits), dtype=int))
        else:
            bits_ref = bits[:len(bits_hat)]

        erros = np.sum(bits_ref != bits_hat)
        ber.append(erros / len(bits_ref))

    return np.array(ber)


# ==========================================================
# 5) MAIN - escolha de mensagem e simulação
# ==========================================================

if __name__ == "__main__":
    np.random.seed()

    # Escolha da mensagem
    frase_default = "The quick brown fox jumps over a lazy dog 0123456789"
    print("=== Escolha da mensagem ===")
    print("1 - Digitar uma mensagem")
    print("2 - Usar mensagem padrão")
    escolha = input("Opção (1/2): ").strip()

    if escolha == "1":
        mensagem = input("Digite a mensagem que deseja transmitir: ")
        if mensagem.strip() == "":
            print("Mensagem vazia, usando a padrão.")
            mensagem = frase_default
    else:
        mensagem = frase_default

    # 1) Gerar bits da mensagem
    bits_mensagem = texto_para_bits(mensagem)
    print("\nMensagem original:", mensagem)
    print("Total de bits gerados:", len(bits_mensagem))
    print("Bits da mensagem (todos):")
    print(bits_mensagem)

    # Reconstrução direta (teste interno)
    texto_recuperado = bits_para_texto(bits_mensagem)
    print("\nReconstrução direta (sem canal):")
    print(texto_recuperado)

    # 2) Mostrar codificação de linha (NRZ, AMI, Manchester)
    plota_codificacao_linha(bits_mensagem)

    # 3) Bits aleatórios para BER
    N_BITS = 100000
    bits = np.random.randint(0, 2, N_BITS)

    # 4) Menu do tipo de canal
    print("\n=== Escolha do tipo de canal para as simulações ===")
    print("1 - Canal AWGN")
    print("2 - Canal com ruído em rajada")
    print("3 - Comparar AWGN e rajada")
    tipo = input("Opção (1/2/3): ").strip()

    # ======================================================
    # 4.1) Demonstração da mensagem transmitida - BPSK
    # ======================================================
    EbN0_demo = 4.0
    print(f"\n=== Demonstração da mensagem transmitida ===")
    print(f"Modulação: BPSK | Eb/N0 = {EbN0_demo} dB")

    s_msg = bpsk_mod(bits_mensagem)

    if tipo == "1":
        r_msg = canal_ruidoso(s_msg, EbN0_demo, bits_por_simbolo=1, tipo_ruido="awgn")
        descricao_canal_demo = "AWGN"
    elif tipo == "2":
        r_msg = canal_ruidoso(s_msg, EbN0_demo, bits_por_simbolo=1,
                              tipo_ruido="rajada", burst_prob=0.02, burst_factor=20)
        descricao_canal_demo = "ruído em rajada"
    else:
        r_msg = canal_ruidoso(s_msg, EbN0_demo, bits_por_simbolo=1, tipo_ruido="awgn")
        descricao_canal_demo = "AWGN (comparação)"

    bits_msg_rx = bpsk_demod(r_msg)
    bits_msg_rx = bits_msg_rx[:len(bits_mensagem)]
    texto_rx = bits_para_texto(bits_msg_rx)

    print(f"Canal usado: {descricao_canal_demo}")
    print("Mensagem recebida via BPSK:")
    print(texto_rx)

    # ======================================================
    # 4.2) Demonstração da mensagem transmitida - QPSK
    # ======================================================

    print(f"\nModulação: QPSK | Eb/N0 = {EbN0_demo} dB")

    s_msg_qpsk, n_bits_total = qpsk_mod(bits_mensagem)

    if tipo == "1":
        r_msg_qpsk = canal_ruidoso(s_msg_qpsk, EbN0_demo, bits_por_simbolo=2,
                                   tipo_ruido="awgn")
    elif tipo == "2":
        r_msg_qpsk = canal_ruidoso(s_msg_qpsk, EbN0_demo, bits_por_simbolo=2,
                                   tipo_ruido="rajada", burst_prob=0.02, burst_factor=20)
    else:
        r_msg_qpsk = canal_ruidoso(s_msg_qpsk, EbN0_demo, bits_por_simbolo=2,
                                   tipo_ruido="awgn")

    bits_msg_rx_qpsk = qpsk_demod(r_msg_qpsk, n_bits_total)
    bits_msg_rx_qpsk = bits_msg_rx_qpsk[:len(bits_mensagem)]

    texto_rx_qpsk = bits_para_texto(bits_msg_rx_qpsk)

    print("Mensagem recebida via QPSK:")
    print(texto_rx_qpsk)

    # ======================================================
    # 5) Simulação de BER
    # ======================================================

    EbN0_dBs = np.arange(0, 11, 1)

    if tipo == "1":
        # APENAS AWGN
        ber_bpsk = simula_ber_bpsk(bits, EbN0_dBs, tipo_ruido="awgn")
        ber_qpsk = simula_ber_qpsk(bits, EbN0_dBs, tipo_ruido="awgn")

        plt.figure(figsize=(9, 6))
        plt.semilogy(EbN0_dBs, ber_bpsk, 'o-', label='BPSK - AWGN')
        plt.semilogy(EbN0_dBs, ber_qpsk, 's-', label='QPSK - AWGN')

        plt.grid(True, which='both')
        plt.xlabel('Eb/N0 (dB)')
        plt.ylabel('BER')
        plt.title('BER x Eb/N0 - Canal AWGN')
        plt.legend()
        plt.tight_layout()
        plt.show()

    elif tipo == "2":
        # APENAS RAJADA
        ber_bpsk = simula_ber_bpsk(bits, EbN0_dBs,
                                   tipo_ruido="rajada", burst_prob=0.02, burst_factor=20)
        ber_qpsk = simula_ber_qpsk(bits, EbN0_dBs,
                                   tipo_ruido="rajada", burst_prob=0.02, burst_factor=20)

        plt.figure(figsize=(9, 6))
        plt.semilogy(EbN0_dBs, ber_bpsk, 'o--', label='BPSK - rajada')
        plt.semilogy(EbN0_dBs, ber_qpsk, 's--', label='QPSK - rajada')

        plt.grid(True, which='both')
        plt.xlabel('Eb/N0 (dB)')
        plt.ylabel('BER')
        plt.title('BER x Eb/N0 - Canal com ruído em rajada')
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        # COMPARAÇÃO AWGN x RAJADA
        ber_bpsk_awgn = simula_ber_bpsk(bits, EbN0_dBs, tipo_ruido="awgn")
        ber_qpsk_awgn = simula_ber_qpsk(bits, EbN0_dBs, tipo_ruido="awgn")

        ber_bpsk_raj = simula_ber_bpsk(bits, EbN0_dBs,
                                       tipo_ruido="rajada", burst_prob=0.02, burst_factor=20)
        ber_qpsk_raj = simula_ber_qpsk(bits, EbN0_dBs,
                                       tipo_ruido="rajada", burst_prob=0.02, burst_factor=20)

        plt.figure(figsize=(9, 6))
        plt.semilogy(EbN0_dBs, ber_bpsk_awgn, 'o-', label='BPSK - AWGN')
        plt.semilogy(EbN0_dBs, ber_qpsk_awgn, 's-', label='QPSK - AWGN')
        plt.semilogy(EbN0_dBs, ber_bpsk_raj, 'o--', label='BPSK - rajada')
        plt.semilogy(EbN0_dBs, ber_qpsk_raj, 's--', label='QPSK - rajada')

        plt.grid(True, which='both')
        plt.xlabel('Eb/N0 (dB)')
        plt.ylabel('BER')
        plt.title('BER x Eb/N0\nBPSK e QPSK - AWGN vs. Rajada')
        plt.legend()
        plt.tight_layout()
        plt.show()

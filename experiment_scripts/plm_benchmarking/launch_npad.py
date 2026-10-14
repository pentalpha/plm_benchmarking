import os
import time
import subprocess
import logging
from pathlib import Path

# Configuração de log para saídas elegantes no terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ---------------------------------------------------------
# CONSTANTES E CONFIGURAÇÕES
# ---------------------------------------------------------
N_COMBINATIONS = 64
N_TARGETS = 32
MIN_ANNOTATIONS = 120
MAX_TRAIN_PROTEINS = 80000

# Usando pathlib para manipular caminhos de forma robusta
BASE_RESULTS_DIR = Path("outputs/plm_benchmarking")
EMBS_DIR = Path("/home/pdaasobrinho/scratch/release_2")
LIGHT_TEMPLATE_PATH = "experiment_scripts/plm_benchmarking/gpu_template_light.slurm"
HEAVY_TEMPLATE_PATH = "experiment_scripts/plm_benchmarking/gpu_template_heavy.slurm"

# Define o tempo de espera entre as checagens (em segundos) - Ex: 10 minutos
CHECK_INTERVAL_SECONDS = 40

# Pega o usuário logado no SO dinamicamente 
# (squeue geralmente trunca para 8 caracteres: 'pdaasobr')
USER = os.environ.get("USER", "pdaasobrinho")
USER_SHORT = USER[:8] 

# Modelos e seus respectivos templates

CONFIGS = [
    ("esmc_300", LIGHT_TEMPLATE_PATH)
]

'''("esm2_150", LIGHT_TEMPLATE_PATH),
    ("e1_300", LIGHT_TEMPLATE_PATH),
    ("esm2_650", LIGHT_TEMPLATE_PATH),
    ("e1_600", LIGHT_TEMPLATE_PATH),
    ("esmc_600", LIGHT_TEMPLATE_PATH),
    ("ankh_base", LIGHT_TEMPLATE_PATH),
    ("ankh2_large", LIGHT_TEMPLATE_PATH),
    ("ankh3_large_nlu", LIGHT_TEMPLATE_PATH),
    ("amplify_350", LIGHT_TEMPLATE_PATH),
    ("amplify_120", LIGHT_TEMPLATE_PATH),
    ("e1_150", LIGHT_TEMPLATE_PATH),
    ("ankh_large", LIGHT_TEMPLATE_PATH),'''
'''("ankh3_large_s2s", LIGHT_TEMPLATE_PATH),
    ("ankh3_large", LIGHT_TEMPLATE_PATH),
    '''

# ---------------------------------------------------------
# FUNÇÕES DE CONTROLE
# ---------------------------------------------------------
def get_jobs_status() -> dict:
    """
    Executa o squeue filtrando por usuário e formatando a saída para retornar 
    apenas Nome do Job e Estado (R, PD). Retorna um dicionário {nome_do_job: estado}.
    """
    try:
        # -h oculta o cabeçalho
        # -o "%50j %t" pega o nome do job (com padding para não cortar) e o status do job
        result = subprocess.run(
            ["squeue", "-u", USER, "-h", "-o", "%50j %t"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            universal_newlines=True, 
            check=True
        )
        
        active_jobs = {}
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                job_name = parts[0]
                job_state = parts[1]
                active_jobs[job_name] = job_state
        return active_jobs
    except subprocess.CalledProcessError as e:
        logging.error(f"[Erro] Falha ao consultar o squeue: {e}")
        return None 

def get_progress_count(model_name: str) -> int:
    """
    Entra no diretório do modelo e conta quantos 'results_eval.json' 
    existem dentro das subpastas 'model_*'.
    """
    model_dir = BASE_RESULTS_DIR / model_name
    if not model_dir.is_dir():
        return 0
        
    count = 0
    # Itera sobre os itens do diretório do modelo
    for item in model_dir.iterdir():
        # Se for um diretório e começar com 'model_'
        if item.is_dir() and item.name.startswith("model_"):
            # Verifica se o json de sucesso existe lá dentro
            if (item / "results_eval.json").is_file():
                count += 1
    return count

def print_status_board():
    active_jobs = get_jobs_status()
    
    if active_jobs is None:
        return False, 999, []

    print("\n" + "="*65)
    print(f"📊 STATUS DAS OTIMIZAÇÕES - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65)
    
    all_done = True
    active_count = 0
    incomplete_models = []

    for model_name, template in CONFIGS:
        result_tsv = BASE_RESULTS_DIR / model_name / f"optimized_results_{N_COMBINATIONS}.tsv"
        
        # Calcula o progresso real mapeando as pastas com sucesso
        progress = get_progress_count(model_name)
        # Formata com zero à esquerda (ex: 05 de 48) para manter tudo alinhado
        prog_str = f"({progress:02d} de {N_COMBINATIONS})"
        
        model_str = f"{model_name:<15}"

        enough_coms = progress >= N_COMBINATIONS
        
        if enough_coms:
            print(f"[✓] {model_str} : Concluído       {prog_str}")
        elif model_name in active_jobs:
            all_done = False
            active_count += 1
            state = active_jobs[model_name]
            
            if state == "R":
                print(f"[▶] {model_str} : Em Execução     {prog_str}")
            elif state == "PD":
                print(f"[⏸] {model_str} : Na Fila (Slurm) {prog_str}")
            elif state == "CG":
                print(f"[⌛] {model_str} : Finalizando     {prog_str}")
            else:
                print(f"[⚙] {model_str} : Status: {state:<7} {prog_str}")
        else:
            all_done = False
            incomplete_models.append((model_name, template, progress))
            print(f"[⏳] {model_str} : Aguarda Submis. {prog_str}")
    incomplete_models.sort(key=lambda x: x[2])
    incomplete_models = [(a,b) for a,b,c in incomplete_models]
    print("="*65 + "\n")
    return all_done, active_count, incomplete_models

def submit_job(model_name: str, template_path: str):
    """
    Submete um job individual ao Slurm montando o comando sbatch.
    """
    out_dir = BASE_RESULTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_prefix = EMBS_DIR / f"emb.{model_name}"

    cmd = [
        "sbatch", f"--job-name={model_name}", template_path,
        str(N_TARGETS), str(MIN_ANNOTATIONS), str(MAX_TRAIN_PROTEINS),
        str(parquet_prefix), str(out_dir), str(N_COMBINATIONS)
    ]

    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, universal_newlines=True
        )
        print(f"    ↳ 🚀 Submetido: {model_name}")
    except subprocess.CalledProcessError as e:
        print(f"    ↳ ❌ Falha ao submeter {model_name}. Erro: {e.stderr.strip()}")

# ---------------------------------------------------------
# LOOP PRINCIPAL
# ---------------------------------------------------------
if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    
    # Limpa a tela na primeira vez que rodar para ficar bonito no screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    while True:
        # 1. Imprime o painel e recebe os status atuais
        all_done, active_jobs_count, incomplete_models = print_status_board()
        
        # Se estiver tudo concluído, finaliza o script
        if all_done:
            print("🎉 Todos os 48 alvos dos modelos foram otimizados com sucesso!")
            print("Encerrando o orquestrador.\n")
            break
        elif active_jobs_count == 999:
            #error getting infos
            print("❌ Erro ao obter status dos jobs. Tentando novamente em 10 segundos...")
            time.sleep(10)
            continue
            
        # 2. Se a fila do usuário secou, dispara a próxima rodada
        if active_jobs_count == 0 and incomplete_models:
            print(f"🔄 Fila vazia! Disparando nova rodada para {len(incomplete_models)} modelos:\n")
            for model_name, template in incomplete_models:
                submit_job(model_name, template)
            print(f"\n💤 Rodada concluída. Aguardando próximo ciclo de checagem...")
        else:
            print(f"💤 Processos em andamento. Aguardando próximo ciclo...")

        # 3. Pausa até a próxima checagem
        time.sleep(CHECK_INTERVAL_SECONDS)
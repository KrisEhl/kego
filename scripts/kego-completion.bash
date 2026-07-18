_kego() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Find where "kego" is in the command line words (offset-aware)
    local kego_idx=-1
    for ((i=0; i<=COMP_CWORD; i++)); do
        if [ "${COMP_WORDS[i]}" = "kego" ]; then
            kego_idx=$i
            break
        fi
    done

    # If "kego" is not found in COMP_WORDS, default to index 0
    if [ $kego_idx -eq -1 ]; then
        kego_idx=0
    fi

    # Subcommands
    local subcommands="run ensemble tune submit status cache submissions league battle train-agent models"

    # Global options
    local global_opts="--config --params --executor --force --tag --task"

    # The subcommand itself is the word immediately following "kego"
    local subcmd="${COMP_WORDS[kego_idx+1]}"

    # Command-specific options
    case "${subcmd}" in
        run)
            opts="--fast --no-ensemble --submit --model --hp-tune --hp-params"
            ;;
        ensemble)
            opts="--from-experiment --from-ensemble"
            ;;
        tune)
            opts="--tune --trials"
            ;;
        submit)
            opts="--from-ensemble --message"
            ;;
        cache)
            opts="status prune"
            ;;
        league)
            opts="--games --search-count --workers --debug --cache-dir --write-ratings --include-local-mcts --partial-save-every --target matrix merge"
            ;;
        battle)
            opts="--agent1 --agent2 --games --deck1 --deck2"
            ;;
        train-agent)
            opts="--agent --epochs --output --init-checkpoint --num-workers --variant --target"
            ;;
        models)
            case "$prev" in
                prune)
                    opts="--drop-worse --drop-worse-min-games --drop-worse-k"
                    ;;
                *)
                    # Check if 'prune' was typed after the "models" subcmd
                    local is_prune=false
                    for ((i=kego_idx+2; i<COMP_CWORD; i++)); do
                        if [ "${COMP_WORDS[i]}" = "prune" ]; then
                            is_prune=true
                            break
                        fi
                    done
                    if [ "$is_prune" = true ]; then
                        opts="--drop-worse --drop-worse-min-games --drop-worse-k"
                    else
                        opts="--sort-by --breakdown --color prune unprune"
                    fi
                    ;;
            esac
            ;;
        *)
            # If the cursor is right after "kego", offer subcommands
            if [ $COMP_CWORD -eq $((kego_idx+1)) ]; then
                opts="$subcommands"
            else
                opts="$subcommands $global_opts"
            fi
            ;;
    esac

    COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
    return 0
}
complete -F _kego kego

MODULE useapplu

    USE lu, ONLY: dolu

CONTAINS

    SUBROUTINE call_dolu()
        CALL dolu()
    END SUBROUTINE call_dolu

END MODULE useapplu
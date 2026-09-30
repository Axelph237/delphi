from __future__ import annotations

import io

import pytest

from cstcompiler import compiler
from cstcompiler.backbone import max_covered_leaves
from cstcompiler.compiler import CST, VariableMapping, clauses_from_dimacs
from cstcompiler.numpy_context import build_numpy_context
from cstcompiler.synthesis import Clause

from tests.utils import cnf_truth_mask, input_masks, simulate


# ---------- Helpers ----------

def _assert_oracle_computes(cst: CST, clauses: list[Clause]) -> None:
    """Exhaustively check the compiled oracle against f on every input and both target values."""
    circuit, x_register, [target] = cst.to_oracle()
    ctx = cst.ctx

    masks = input_masks(ctx.n_vars)
    all_ones = 2 ** (2 ** ctx.n_vars) - 1
    f = cnf_truth_mask(clauses, {v: masks[i] for v, i in ctx.var_to_idx.items()}, all_ones)

    assert x_register == list(range(ctx.n_vars))
    ancilla_count = circuit.num_qubits - ctx.n_vars
    for c in (0, all_ones):
        wires = masks + [0] * ancilla_count
        wires[target] = c
        result = simulate(circuit, wires, all_ones)

        expected = masks + [0] * ancilla_count
        expected[target] = c ^ f
        assert result == expected


def _clauses(*literal_groups: tuple[int, ...]) -> list[Clause]:
    return [Clause.from_literals(group) for group in literal_groups]


# ---------- VariableMapping ----------

class TestVariableMappingDefaults:
    def test__VariableMapping__defaults_are_empty_and_ids_start_at_one(self):
        vm = VariableMapping()
        assert vm.variable_ids == {}
        assert vm.variable_names == {}
        assert vm.next_var_id == 1

    def test__VariableMapping__instances_do_not_share_dicts(self):
        first, second = VariableMapping(), VariableMapping()
        first.add("x")
        assert second.variable_ids == {} and second.variable_names == {}

    def test__VariableMapping__accepts_explicit_state(self):
        vm = VariableMapping({"a": 7}, {7: "a"}, 8)
        assert (vm.variable_ids, vm.variable_names, vm.next_var_id) == ({"a": 7}, {7: "a"}, 8)

    def test__VariableMapping__compares_by_value(self):
        assert VariableMapping() == VariableMapping()
        assert VariableMapping({"a": 1}, {1: "a"}, 2) != VariableMapping()


class TestVariableMappingAdd:
    def test__add__assigns_sequential_ids_from_one(self):
        vm = VariableMapping()
        for name in ("a", "b", "c"):
            vm.add(name)
        assert vm.variable_ids == {"a": 1, "b": 2, "c": 3}
        assert vm.variable_names == {1: "a", 2: "b", 3: "c"}
        assert vm.next_var_id == 4

    def test__add__never_assigns_zero_so_every_id_can_be_negated(self):
        vm = VariableMapping()
        vm.add("only")
        assert 0 not in vm.variable_names
        assert -vm.variable_ids["only"] < 0

    def test__add__same_name_twice_rebinds_the_name_and_orphans_the_old_id(self):
        # Current behavior: `add` does not check for an existing name, so the old
        # id survives in variable_names with no inverse entry.
        vm = VariableMapping()
        vm.add("x")
        vm.add("x")
        assert vm.variable_ids == {"x": 2}
        assert vm.variable_names == {1: "x", 2: "x"}


class TestVariableMappingRemove:
    def test__remove__by_name_drops_both_directions(self):
        vm = VariableMapping()
        vm.add("a")
        vm.add("b")
        vm.remove("a")
        assert vm.variable_ids == {"b": 2}
        assert vm.variable_names == {2: "b"}

    def test__remove__by_id_drops_both_directions(self):
        vm = VariableMapping()
        vm.add("a")
        vm.add("b")
        vm.remove(1)
        assert vm.variable_ids == {"b": 2}
        assert vm.variable_names == {2: "b"}

    def test__remove__does_not_recycle_the_freed_id(self):
        vm = VariableMapping()
        for name in ("a", "b", "c"):
            vm.add(name)
        vm.remove("b")
        vm.add("d")
        assert vm.variable_ids == {"a": 1, "c": 3, "d": 4}
        assert vm.variable_names[vm.variable_ids["c"]] == "c"

    def test__remove__unknown_name_raises_key_error(self):
        with pytest.raises(KeyError):
            VariableMapping().remove("missing")

    def test__remove__unknown_id_raises_key_error(self):
        with pytest.raises(KeyError):
            VariableMapping().remove(99)

    def test__remove__id_equal_to_a_float_is_treated_as_that_id(self):
        # Not a str, so it takes the id branch, where 1.0 hashes equal to 1.
        vm = VariableMapping()
        vm.add("a")
        vm.remove(1.0)  # type: ignore[arg-type]
        assert vm.variable_ids == {} and vm.variable_names == {}

    def test__remove__twice_raises_key_error(self):
        vm = VariableMapping()
        vm.add("a")
        vm.remove("a")
        with pytest.raises(KeyError):
            vm.remove("a")

    def test__remove__boolean_key_is_treated_as_its_integer_id(self):
        vm = VariableMapping()
        vm.add("a")
        vm.remove(True)
        assert vm.variable_ids == {} and vm.variable_names == {}

    def test__add__accepts_the_empty_name(self):
        vm = VariableMapping()
        vm.add("")
        assert vm.variable_ids == {"": 1}

    def test__remove__by_name_after_a_rebind_leaves_the_orphaned_id(self):
        # The name resolves to the newer id, so the older id survives with no inverse.
        vm = VariableMapping()
        vm.add("x")
        vm.add("x")
        vm.remove("x")
        assert vm.variable_ids == {}
        assert vm.variable_names == {1: "x"}

    def test__remove__by_stale_id_drops_the_live_inverse_entry(self):
        # After a rebind the name resolves to the newer id, so removing the older
        # id deletes the surviving name entry and leaves the newer id inverse-less.
        vm = VariableMapping()
        vm.add("x")
        vm.add("x")
        vm.remove(1)
        assert vm.variable_ids == {}
        assert vm.variable_names == {2: "x"}


# ---------- clauses_from_dimacs ----------

class TestClausesFromDimacsSource:
    def test__clauses_from_dimacs__accepts_a_string(self):
        assert clauses_from_dimacs("p cnf 2 1\n1 -2 0\n") == _clauses((1, -2))

    def test__clauses_from_dimacs__accepts_a_text_stream(self):
        source = io.StringIO("p cnf 2 1\n1 -2 0\n")
        assert clauses_from_dimacs(source) == _clauses((1, -2))

    def test__clauses_from_dimacs__builds_its_own_mapping_when_none_is_given(self):
        assert clauses_from_dimacs("c var 1 : x\np cnf 1 1\n1 0\n") == _clauses((1,))

    def test__clauses_from_dimacs__fills_a_supplied_mapping(self):
        vm = VariableMapping()
        clauses_from_dimacs("c var 1 : x\nc var 2 : y\np cnf 2 1\n1 -2 0\n", vm)
        assert vm.variable_ids == {"x": 1, "y": 2}
        assert vm.variable_names == {1: "x", 2: "y"}


class TestClausesFromDimacsComments:
    def test__clauses_from_dimacs__ignores_comments_that_are_not_var_declarations(self):
        vm = VariableMapping()
        assert clauses_from_dimacs("c hello\nc\np cnf 1 1\n1 0\n", vm) == _clauses((1,))
        assert vm.variable_ids == {}

    def test__clauses_from_dimacs__keeps_names_containing_spaces(self):
        vm = VariableMapping()
        clauses_from_dimacs("c var 1 : alice is red\np cnf 1 1\n1 0\n", vm)
        assert vm.variable_ids == {"alice is red": 1}

    def test__clauses_from_dimacs__records_declarations_appearing_after_clauses(self):
        vm = VariableMapping()
        clauses_from_dimacs("p cnf 1 1\n1 0\nc var 1 : late\n", vm)
        assert vm.variable_names == {1: "late"}

    def test__clauses_from_dimacs__records_a_zero_variable_id_verbatim(self):
        vm = VariableMapping()
        clauses_from_dimacs("c var 0 : zero\np cnf 1 1\n1 0\n", vm)
        assert vm.variable_names == {0: "zero"}

    def test__clauses_from_dimacs__leaves_undeclared_variables_out_of_the_mapping(self):
        vm = VariableMapping()
        clauses_from_dimacs("c var 1 : x\np cnf 2 1\n1 2 0\n", vm)
        assert vm.variable_ids == {"x": 1}


class TestClausesFromDimacsClauses:
    def test__clauses_from_dimacs__skips_blank_and_whitespace_only_lines(self):
        assert clauses_from_dimacs("p cnf 1 1\n\n   \n1 0\n") == _clauses((1,))

    def test__clauses_from_dimacs__parses_a_bare_terminator_as_the_empty_clause(self):
        assert clauses_from_dimacs("p cnf 1 1\n0\n") == [Clause(set(), 0)]

    def test__clauses_from_dimacs__parses_negative_literals(self):
        assert clauses_from_dimacs("p cnf 3 2\n-1 2 0\n-3 0\n") == _clauses((-1, 2), (-3,))

    def test__clauses_from_dimacs__tolerates_extra_whitespace_between_literals(self):
        assert clauses_from_dimacs("p cnf 2 1\n  1    -2   0  \n") == _clauses((1, -2))

    def test__clauses_from_dimacs__header_with_zero_counts_yields_no_clauses(self):
        assert clauses_from_dimacs("p cnf 0 0\n") == []

    def test__clauses_from_dimacs__ignores_header_counts_that_disagree_with_the_body(self):
        assert clauses_from_dimacs("p cnf 99 99\n1 0\n") == _clauses((1,))

    def test__clauses_from_dimacs__accepts_a_repeated_header(self):
        assert clauses_from_dimacs("p cnf 1 1\np cnf 2 2\n1 0\n") == _clauses((1,))

    def test__clauses_from_dimacs__collapses_a_repeated_literal(self):
        assert clauses_from_dimacs("p cnf 1 1\n1 1 0\n") == _clauses((1,))

    def test__clauses_from_dimacs__accepts_any_iterable_of_lines(self):
        assert clauses_from_dimacs(["p cnf 2 1\n", "1 -2 0\n"]) == _clauses((1, -2))  # type: ignore[arg-type]

    def test__clauses_from_dimacs__handles_windows_line_endings(self):
        assert clauses_from_dimacs("p cnf 2 1\r\n1 -2 0\r\n") == _clauses((1, -2))

    def test__clauses_from_dimacs__accepts_tab_separated_literals(self):
        assert clauses_from_dimacs("p cnf 2 1\n1\t-2\t0\n") == _clauses((1, -2))

    def test__clauses_from_dimacs__accepts_an_explicit_plus_sign(self):
        assert clauses_from_dimacs("p cnf 1 1\n+1 0\n") == _clauses((1,))

    def test__clauses_from_dimacs__reads_negative_zero_as_variable_zero(self):
        # -0 == 0, so the sign is lost and the literal lands on variable 0.
        assert clauses_from_dimacs("p cnf 1 1\n-0 0\n") == [Clause({0}, 0)]

    def test__clauses_from_dimacs__does_not_advance_the_id_counter(self):
        # Declarations are written straight into both dicts, so next_var_id still
        # points at 1 and a later add() hands out an id DIMACS already claimed.
        vm = VariableMapping()
        clauses_from_dimacs("c var 1 : x\nc var 2 : y\np cnf 2 1\n1 2 0\n", vm)
        assert vm.next_var_id == 1

        vm.add("fresh")
        assert vm.variable_ids == {"x": 1, "y": 2, "fresh": 1}
        assert vm.variable_names == {1: "fresh", 2: "y"}

    def test__clauses_from_dimacs__one_id_declared_with_two_names_keeps_the_last(self):
        vm = VariableMapping()
        clauses_from_dimacs("c var 1 : x\nc var 1 : y\np cnf 1 1\n1 0\n", vm)
        assert vm.variable_ids == {"x": 1, "y": 1}
        assert vm.variable_names == {1: "y"}

    def test__clauses_from_dimacs__one_name_declared_at_two_ids_keeps_the_last(self):
        # variable_ids holds one entry while variable_names keeps both, so the
        # mapping is no longer a bijection.
        vm = VariableMapping()
        clauses_from_dimacs("c var 1 : x\nc var 2 : x\np cnf 2 1\n1 0\n", vm)
        assert vm.variable_ids == {"x": 2}
        assert vm.variable_names == {1: "x", 2: "x"}


class TestClausesFromDimacsErrors:
    @pytest.mark.parametrize("header", ["p cnf 3", "p cnf", "p cnf 3 3 3", "p cnf a b", "p"])
    def test__clauses_from_dimacs__malformed_header_raises(self, header):
        with pytest.raises(ValueError, match="Malformed header line"):
            clauses_from_dimacs(f"{header}\n")

    def test__clauses_from_dimacs__clause_before_header_raises(self):
        with pytest.raises(ValueError, match="Encountered clause line before header"):
            clauses_from_dimacs("1 -2 0\n")

    def test__clauses_from_dimacs__uppercase_header_is_treated_as_a_clause_line(self):
        with pytest.raises(ValueError, match="Encountered clause line before header"):
            clauses_from_dimacs("P CNF 1 1\n1 0\n")

    def test__clauses_from_dimacs__unterminated_clause_raises(self):
        with pytest.raises(ValueError, match="not terminated by 0"):
            clauses_from_dimacs("p cnf 2 1\n1 -2\n")

    def test__clauses_from_dimacs__content_after_the_terminator_raises(self):
        with pytest.raises(ValueError, match="not terminated by 0"):
            clauses_from_dimacs("p cnf 3 1\n1 2 0 3\n")

    def test__clauses_from_dimacs__non_integer_literal_raises(self):
        with pytest.raises(ValueError, match="invalid literal for int"):
            clauses_from_dimacs("p cnf 1 1\nfoo 0\n")

    def test__clauses_from_dimacs__variable_and_its_negation_in_one_clause_raises(self):
        with pytest.raises(ValueError, match="both 1 and -1"):
            clauses_from_dimacs("p cnf 2 1\n1 -1 2 0\n")


# ---------- CST.from_named_literals ----------

class TestFromNamedLiterals:
    def test__from_named_literals__assigns_ids_in_first_sighting_order(self):
        cst = CST.from_named_literals([["b", "a"], ["c"]], ancilla_budget=8)
        assert cst.variable_mapping is not None
        assert cst.variable_mapping.variable_ids == {"b": 1, "a": 2, "c": 3}

    @pytest.mark.parametrize("prefix", ["~", "-"])
    def test__from_named_literals__both_prefixes_negate_the_base_variable(self, prefix):
        cst = CST.from_named_literals([[f"{prefix}x", "y"]], ancilla_budget=8)
        assert cst.variable_mapping is not None
        assert cst.variable_mapping.variable_ids == {"x": 1, "y": 2}
        [clause] = [c for batch in cst.root.partition for c in batch.clauses]
        assert clause.literals == (-1, 2)

    def test__from_named_literals__shares_one_id_across_polarities_in_different_clauses(self):
        cst = CST.from_named_literals([["x", "y"], ["~x", "y"]], ancilla_budget=8)
        assert cst.variable_mapping is not None
        assert cst.variable_mapping.variable_ids == {"x": 1, "y": 2}

    def test__from_named_literals__registers_the_base_name_when_first_seen_negated(self):
        cst = CST.from_named_literals([["~x"], ["x", "y"]], ancilla_budget=8)
        assert cst.variable_mapping is not None
        assert cst.variable_mapping.variable_ids == {"x": 1, "y": 2}

    def test__from_named_literals__accepts_generators_for_both_levels(self):
        clauses = (iter(group) for group in [["x", "~y"], ["y"]])
        cst = CST.from_named_literals(clauses, ancilla_budget=8)
        assert cst.variable_mapping is not None
        assert cst.variable_mapping.variable_ids == {"x": 1, "y": 2}

    def test__from_named_literals__collapses_a_repeated_literal_within_a_clause(self):
        cst = CST.from_named_literals([["x", "x"], ["y"]], ancilla_budget=8)
        assert cst.variable_mapping is not None
        assert cst.variable_mapping.variable_ids == {"x": 1, "y": 2}

    def test__from_named_literals__treats_a_bare_prefix_as_the_empty_name(self):
        cst = CST.from_named_literals([["~", "y"]], ancilla_budget=8)
        assert cst.variable_mapping is not None
        assert cst.variable_mapping.variable_ids == {"": 1, "y": 2}

    def test__from_named_literals__accepts_an_empty_clause(self):
        cst = CST.from_named_literals([[], ["x"]], ancilla_budget=8)
        clauses = [c for batch in cst.root.partition for c in batch.clauses]
        assert Clause(set(), 0) in clauses

    def test__from_named_literals__variable_and_its_negation_in_one_clause_raises(self):
        with pytest.raises(ValueError, match="both 1 and -1"):
            CST.from_named_literals([["x", "~x"]], ancilla_budget=8)

    def test__from_named_literals__no_clauses_raises(self):
        with pytest.raises(RuntimeError, match="Cannot implement CST"):
            CST.from_named_literals([], ancilla_budget=8)

    def test__from_named_literals__budget_too_small_raises(self):
        with pytest.raises(ValueError, match="Cannot implement more than"):
            CST.from_named_literals([["a"], ["b"], ["c"], ["d"], ["e"]], ancilla_budget=2)

    @pytest.mark.parametrize("name", ["~~x", "--x", "-~x"])
    def test__from_named_literals__strips_only_the_outermost_prefix(self, name):
        cst = CST.from_named_literals([[name, "y"]], ancilla_budget=8)
        assert cst.variable_mapping is not None
        assert set(cst.variable_mapping.variable_ids) == {name[1:], "y"}

    @pytest.mark.parametrize("name", ["a-b", "a~b", "x-"])
    def test__from_named_literals__only_a_leading_prefix_negates(self, name):
        cst = CST.from_named_literals([[name, "y"]], ancilla_budget=8)
        assert cst.variable_mapping is not None
        assert set(cst.variable_mapping.variable_ids) == {name, "y"}
        [clause] = [c for batch in cst.root.partition for c in batch.clauses]
        assert clause.variable_polarity_mask == 0

    def test__from_named_literals__empty_generator_raises(self):
        with pytest.raises(RuntimeError, match="Cannot implement CST"):
            CST.from_named_literals((group for group in []), ancilla_budget=8)

    def test__from_named_literals__only_empty_clauses_declare_no_variables(self):
        cst = CST.from_named_literals([[], []], ancilla_budget=8)
        assert cst.variable_mapping is not None
        assert cst.variable_mapping.variable_ids == {}
        assert cst.ctx.n_vars == 0

    def test__from_named_literals__non_string_literal_raises(self):
        with pytest.raises(AttributeError, match="startswith"):
            CST.from_named_literals([[1, 2]], ancilla_budget=8)  # type: ignore[list-item]


# ---------- CST.from_dimacs ----------

class TestFromDimacs:
    DIMACS = "c var 1 : x\nc var 2 : y\np cnf 3 3\n1 -2 0\n2 3 0\n-3 0\n"

    def test__from_dimacs__populates_both_directions_of_the_mapping(self):
        cst = CST.from_dimacs(self.DIMACS, ancilla_budget=8)
        assert cst.variable_mapping is not None
        assert cst.variable_mapping.variable_ids == {"x": 1, "y": 2}
        assert cst.variable_mapping.variable_names == {1: "x", 2: "y"}

    def test__from_dimacs__compiles_every_parsed_clause(self):
        cst = CST.from_dimacs(self.DIMACS, ancilla_budget=8)
        clauses = {c.literals for batch in cst.root.partition for c in batch.clauses}
        assert clauses == {(1, -2), (2, 3), (-3,)}

    def test__from_dimacs__uses_the_requested_ancilla_budget(self):
        cst = CST.from_dimacs(self.DIMACS, ancilla_budget=9)
        assert cst.root.size == 9

    def test__from_dimacs__empty_document_raises(self):
        with pytest.raises(RuntimeError, match="Cannot implement CST"):
            CST.from_dimacs("p cnf 0 0\n", ancilla_budget=8)

    def test__from_dimacs__propagates_parse_errors(self):
        with pytest.raises(ValueError, match="Malformed header line"):
            CST.from_dimacs("p cnf bad\n", ancilla_budget=8)

    def test__from_dimacs__accepts_a_text_stream(self):
        cst = CST.from_dimacs(io.StringIO("p cnf 1 1\n1 0\n"), ancilla_budget=8)  # type: ignore[arg-type]
        assert cst.ctx.n_vars == 1

    def test__from_dimacs__document_of_only_empty_clauses_has_no_variables(self):
        cst = CST.from_dimacs("p cnf 0 2\n0\n0\n", ancilla_budget=8)
        assert cst.ctx.n_vars == 0


# ---------- CST.from_Clauses ----------

class TestFromClauses:
    def test__from_Clauses__defaults_the_variable_mapping_to_none(self):
        cst = CST.from_Clauses(_clauses((1, 2), (2, 3)), ancilla_budget=8)
        assert cst.variable_mapping is None

    def test__from_Clauses__keeps_a_supplied_mapping(self):
        vm = VariableMapping()
        vm.add("x")
        cst = CST.from_Clauses(_clauses((1, 2)), ancilla_budget=8, var_map=vm)
        assert cst.variable_mapping is vm

    def test__from_Clauses__mirrors_the_asdt_root_and_exposes_its_context(self):
        clauses = _clauses((1, 2), (2, 3), (3, 4))
        cst = CST.from_Clauses(clauses, ancilla_budget=8)
        assert cst.root.size == 8
        assert cst.root.depth == 0
        assert cst.root.parent is None
        assert cst.ctx.n_vars == 4

    def test__from_Clauses__sorts_the_caller_list_in_place(self):
        # grow_cst sorts the list it is handed by conflict degree, which the caller observes.
        clauses = _clauses((1, 2, 3), (4,), (1, 4))
        original = list(clauses)
        CST.from_Clauses(clauses, ancilla_budget=8)
        assert clauses != original, "expected the caller's list to be reordered"
        assert [c.normed_variables for c in clauses] == [(4,), (1, 2, 3), (1, 4)]
        assert sorted(clauses, key=lambda c: c.normed_variables) == sorted(original, key=lambda c: c.normed_variables)

    def test__from_Clauses__no_clauses_raises(self):
        with pytest.raises(RuntimeError, match="bad given clauses"):
            CST.from_Clauses([], ancilla_budget=8)

    def test__from_Clauses__budget_too_small_raises(self):
        with pytest.raises(ValueError, match="Cannot implement more than"):
            CST.from_Clauses(_clauses((1,), (2,), (3,), (4,), (5,)), ancilla_budget=2)

    def test__from_Clauses__accepts_exactly_the_ancilla_capacity(self):
        budget = 4
        clauses = _clauses(*[(i,) for i in range(1, max_covered_leaves(budget) + 1)])
        assert CST.from_Clauses(clauses, ancilla_budget=budget).root.size == budget

    def test__from_Clauses__rejects_one_clause_beyond_the_capacity(self):
        budget = 4
        clauses = _clauses(*[(i,) for i in range(1, max_covered_leaves(budget) + 2)])
        with pytest.raises(ValueError, match="Cannot implement more than"):
            CST.from_Clauses(clauses, ancilla_budget=budget)

    def test__from_Clauses__negative_budget_raises(self):
        with pytest.raises(ValueError, match="must have a positive size"):
            CST.from_Clauses(_clauses((1,)), ancilla_budget=-1)

    def test__from_Clauses__zero_budget_builds_a_tree_with_no_ancilla(self):
        cst = CST.from_Clauses(_clauses((1,)), ancilla_budget=0)
        assert cst.root.size == 0
        with pytest.raises(ValueError, match="no allocable ancilla"):
            cst.to_oracle()

    def test__from_Clauses__duplicate_clauses_still_compile_correctly(self):
        clauses = _clauses((1,), (1,), (2,))
        cst = CST.from_Clauses(list(clauses), ancilla_budget=8)
        _assert_oracle_computes(cst, clauses)

    def test__from_Clauses__raises_when_grow_cst_reports_nothing(self, monkeypatch):
        monkeypatch.setattr(compiler, "grow_cst", lambda root, clauses: None)
        with pytest.raises(RuntimeError, match="empty clause list"):
            CST.from_Clauses(_clauses((1,)), ancilla_budget=8)

    def test__from_Clauses__raises_when_grow_cst_yields_no_tree(self, monkeypatch):
        ctx = build_numpy_context(_clauses((1,)))
        monkeypatch.setattr(compiler, "grow_cst", lambda root, clauses: (None, ctx))
        with pytest.raises(RuntimeError, match="returned cst is None"):
            CST.from_Clauses(_clauses((1,)), ancilla_budget=8)

    def test__from_Clauses__raises_when_grow_cst_yields_no_context(self, monkeypatch):
        real_grow_cst = compiler.grow_cst

        def _no_context(root, clauses):
            grown = real_grow_cst(root, clauses)
            assert grown is not None
            return grown[0], None

        monkeypatch.setattr(compiler, "grow_cst", _no_context)
        with pytest.raises(RuntimeError, match="returned cst is None"):
            CST.from_Clauses(_clauses((1,)), ancilla_budget=8)


# ---------- CST.to_oracle ----------

class TestToOracle:
    def test__to_oracle__delegates_the_root_and_context(self, monkeypatch):
        seen = {}

        def _spy(root, ctx):
            seen["root"], seen["ctx"] = root, ctx
            return "circuit", [0], [1]

        monkeypatch.setattr(compiler, "cst_to_oracle", _spy)
        cst = CST.from_Clauses(_clauses((1, 2)), ancilla_budget=8)
        assert cst.to_oracle() == ("circuit", [0], [1])
        assert seen == {"root": cst.root, "ctx": cst.ctx}

    def test__to_oracle__sizes_the_circuit_from_the_variables_and_budget(self):
        cst = CST.from_Clauses(_clauses((1, 2), (2, 3)), ancilla_budget=8)
        circuit, x_register, output_register = cst.to_oracle()
        assert circuit.num_qubits == cst.ctx.n_vars + 8
        assert x_register == [0, 1, 2]
        assert len(output_register) == 1

    def test__to_oracle__is_repeatable(self):
        cst = CST.from_Clauses(_clauses((1, 2), (2, 3)), ancilla_budget=8)
        first, _, first_out = cst.to_oracle()
        second, _, second_out = cst.to_oracle()
        assert first_out == second_out
        assert first.data == second.data

    def test__to_oracle__requires_an_ancilla_for_its_own_target(self):
        with pytest.raises(ValueError, match="too small"):
            CST.from_named_literals([["x"]], ancilla_budget=1).to_oracle()


# ---------- End-to-end oracle correctness ----------

class TestCompiledOracleSemantics:
    @pytest.mark.parametrize("budget", [4, 8])
    def test__from_named_literals__oracle_marks_exactly_the_satisfying_assignments(self, budget):
        cst = CST.from_named_literals([["x", "y"], ["~x", "y"]], ancilla_budget=budget)
        _assert_oracle_computes(cst, _clauses((1, 2), (-1, 2)))

    def test__from_dimacs__oracle_marks_exactly_the_satisfying_assignments(self):
        dimacs = "c var 1 : x\np cnf 3 3\n1 -2 0\n2 3 0\n-3 0\n"
        cst = CST.from_dimacs(dimacs, ancilla_budget=8)
        _assert_oracle_computes(cst, _clauses((1, -2), (2, 3), (-3,)))

    def test__from_Clauses__oracle_of_an_unsatisfiable_formula_marks_nothing(self):
        clauses = _clauses((1,), (-1,))
        cst = CST.from_Clauses(list(clauses), ancilla_budget=8)
        _assert_oracle_computes(cst, clauses)

    def test__from_Clauses__oracle_of_a_tautological_single_clause_marks_everything(self):
        clauses = _clauses((1, -2))
        cst = CST.from_Clauses(list(clauses), ancilla_budget=8)
        _assert_oracle_computes(cst, clauses)

    def test__from_Clauses__oracle_of_only_the_empty_clause_has_no_input_register(self):
        clauses = _clauses(())
        cst = CST.from_Clauses(list(clauses), ancilla_budget=8)
        assert cst.ctx.n_vars == 0
        circuit, x_register, _ = cst.to_oracle()
        assert x_register == []
        _assert_oracle_computes(cst, clauses)

    def test__from_Clauses__oracle_of_an_empty_clause_beside_a_normal_one_marks_nothing(self):
        clauses = _clauses((), (1,))
        cst = CST.from_Clauses(list(clauses), ancilla_budget=8)
        _assert_oracle_computes(cst, clauses)

    def test__from_named_literals__oracle_tracks_polarity_rather_than_name_order(self):
        # "~b" arrives before "b", so a prefix-stripping slip would flip the formula.
        cst = CST.from_named_literals([["~b", "a"], ["b"]], ancilla_budget=8)
        ids = cst.variable_mapping.variable_ids if cst.variable_mapping else {}
        _assert_oracle_computes(cst, _clauses((-ids["b"], ids["a"]), (ids["b"],)))

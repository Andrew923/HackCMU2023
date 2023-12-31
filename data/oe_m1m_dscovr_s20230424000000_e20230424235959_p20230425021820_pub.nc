CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230424000000_e20230424235959_p20230425021820_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-04-25T02:18:20.036Z   date_calibration_data_updated         2023-04-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-04-24T00:00:00.000Z   time_coverage_end         2023-04-24T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx{�@  �          A���A1G���(�@
=@�
=C�T{A1G���(���{��{C�Ff                                    Bx{	�  �          A���AJ�\���@��
A�{C�FfAJ�\��z��E����HC��H                                    Bx{	�  T          A�ffAO33���A'
=A�
=C��qAO33����?�  @�HC���                                    Bx{	"2  �          AŮAZ=q���A1��AظRC�u�AZ=q��33?˅@n�RC��                                    Bx{	0�  
�          A���AA�}��A_\)B�C���AA����@��HA.{C�7
                                    Bx{	?~  t          A�G�A\(��p��Ab{B
�C�:�A\(����
@�z�A?\)C�Ff                                    Bx{	N$  �          A�G�Am��T(�A��
B&�C�%Am���G�A�\A��RC�5�                                    Bx{	\�  �          A�(�A|(��M�A�33B+�RC�p�A|(�����A&ffA�{C���                                    Bx{	kp  T          A�  A���c
=A�  B)�C�nA�����A'�
A�(�C�AH                                    Bx{	z  �          A�p�A�Q��^{A�=qB(�C�8RA�Q���\)A*�RA�{C��                                    Bx{	��  �          A��A�=q�`  A���B)�C�{A�=q����A+33A��
C��f                                    Bx{	�b  T          A�A����n�HA�Q�B
=C���A�����  Ap�A��
C�!H                                    Bx{	�  �          A�
=A�(��r=qA��\BQ�C�aHA�(�����A��A��C��                                    Bx{	��  �          A�RA�z��|z�A�\)B(�C��3A�z����\@�A{
=C��                                    Bx{	�T  �          A�(�A��H���A�z�BQ�C���A��H��
=@��
AIp�C�}q                                    Bx{	��  �          A�{A�\)���A��B��C���A�\)����@��AG33C���                                    Bx{	�  �          A��
A��\��\)A�(�B
�
C�HA��\���\@�(�AH(�C��f                                    Bx{	�F  �          A���A�����G�A�{B��C�AHA�������@�AX��C��\                                    Bx{	��  �          A�\AmG���z�AyG�Bz�C���AmG���G�@�{AG�C���                                    Bx{
�  �          AܸRA>=q����AR�RA�Q�C��3A>=q�ƣ�?�(�@C�
C�˅                                    Bx{
8  �          A��
AP����ffAc\)A�(�C�7
AP������@*=q@��C��)                                    Bx{
)�  �          A�Ah����  A�G�B�C�w
Ah�����@�Q�AIC���                                    Bx{
8�  �          A陚AY�����\A�p�BC��AY����  @�p�A6�HC��                                    Bx{
G*  T          A��A*�R��Q�Ai�A�RC�l�A*�R�ٙ�?�ff@a�C���                                    Bx{
U�  �          A�A{��(�AV�HA�{C��A{��������C��=                                    Bx{
dv  
�          A�Q�@�\)���
AT��Ạ�C���@�\)��׿��ÿ��RC�
=                                    Bx{
s  �          A���@�G���  As\)A��C�
@�G�����?z�H?�C��                                    Bx{
��  �          A��\@�{����A���B	��C��@�{��\@l(�@�33C��f                                    Bx{
�h  �          A�\)AQ���
=A�B0G�C�AQ���ffA	p�A��C�y�                                    Bx{
�  �          A��@�Q���(�AîBQ�C��H@�Q��֏\AT��A�=qC���                                    Bx{
��  T          A�\A1��C�
A߮B�(�C�NA1���
A��BG�C�h�                                    Bx{
�Z  T          A��
A=q@���A�B�33A�A=q�z�A�p�B�\C��q                                    Bx{
�   T          A�z�A�A\)Aԏ\B{�RB*�HA���Q�A�{B���C��{                                    Bx{
٦  �          A�{A33Aw33A�{BC��Bo��A33@w
=A��B���A��R                                    Bx{
�L  �          A�ff@p�A�\)A-p�A��B���@p�A;�A��Bep�B���                                    Bx{
��  
�          A���@�
=A]G�@��A�\)B�(�@�
=A\)AO�
BF��BQ\)                                    Bx{�  �          A���A�
=A\��A^=qB�
B"(�A�
=@�=qA�(�B>�A�Q�                                    Bx{>  	�          A�ffA�G�ABffA?\)A�33B�A�G�@�G�A�Q�B'��A�33                                    Bx{"�            A�AǮAY��@�\)A&�RA�RAǮA(�A0��A��A��H                                    Bx{1�  4          A�=qA��\AV�H?�G�@2�\Ȁ\A��\A1�@�Adz�A���                                    Bx{@0  
�          B
=A�ffA-p��(����  A���A�ffA,��@2�\@�Q�A�(�                                    Bx{N�  	�          B
��B�
A (��أ��4��AZ�\B�
A$(��(��hQ�A�Q�                                    Bx{]|  �          B�HA���@Q��A���@��RA���A	�{�z=qAt��                                    Bx{l"  �          Bz�A��H>\)�H  ��ff>�=qA��H@����/\)���A4��                                    Bx{z�  
�          A�A�����w33���C�7
A��@�ff�k�
���AA��                                    Bx{�n  �          A�=qA�����(��t  ��
C�1�A���?��
���R� ff@�                                      Bx{�            A���A����G��y���\C��{A��׾�(�����4��C�N                                    Bx{��  �          A�  A�ff�b{�s33�	�
C�.A�ff��=q��z��A=qC�W
                                    Bx{�`  
�          A�z�AXz��6�R�yp��%z�C��AXz��0  �����X�RC�AH                                    Bx{�  T          A�=qA_\)�������33C���A_\)�ə���G��]p�C�ٚ                                    Bx{Ҭ  T          A�=qAPz�����m��p�C��HAPz������=q�V�RC�>�                                    Bx{�R  �          A��
AL�����
�j�H��\C��{AL����(����\�X�RC�.                                    Bx{��  �          A��\A%G������!p��مC�޸A%G�� Q�����E=qC��\                                    Bx{��  �          A�
=A*ff�����-���
C���A*ff�(����
=�H�C��H                                    Bx{D  �          A��HA5����ff�  ��C��3A5���-���~�H�5�C�"�                                    Bx{�  �          A��AQ�����R��p���p�C��3AQ���,  �ap����C�P�                                    Bx{*�  �          A�ffA|Q��C33��=q�C�C�!HA|Q��Q��z��ޣ�C�w
                                    Bx{96  h          A�A/
=�PQ���
=����C�fA/
=�	���9G��{C��=                                    Bx{G�  �          AJffAp���p�?fff@��C��Ap���p��9���aG�C���                                    Bx{V�  �          A�=qAff��p��Bff��\)C��RAff�Q���
=�P�C�޸                                    Bx{e(  �          A���@�Q���
=�G\)��Q�C�W
@�Q��n{��\)�Tz�C��)                                    Bx{s�  �          A�=q@e������(��33C�q@e��e���\)�q\)C��                                    Bx{�t  �          BG��8Q����
��ff�#p�C�0��8Q��D������#�C�Q�                                    Bx{�  �          B����ff�����(��&�C��ÿ�ff�=���\)z�C���                                    Bx{��  �          B���^�R��\)��(��5��C�Ǯ�^�R�G�� ���
Cz�                                    Bx{�f  T          B	���
=�������DffC�����
=��(���HCl                                    Bx{�  �          B	\)��������=q�M  C~T{�������G�{Ca}q                                    Bx{˲  T          B�R�33�PQ����
�uQ�Ch�)�33@
=�ff(�C&��                                    Bx{�X  T          B�����K
=���H�y�\Ci����@,���=q��C#ٚ                                    Bx{��  �          B\)� ���;�
��R�{=qCek�� ��@aG�� p�G�C ��                                    Bx{��  �          Bz��(��]p������p�Ck���(�?����RC,(�                                    Bx{J  �          B(�����mp���=q�l��Co.���>���ff��C2Q�                                    Bx{�  �          Bz�����Lz���=q�y�
Cj�����@!��(�C$^�                                    Bx{#�  T          B�
��N�H��\�y{Ck���@���� C%��                                    Bx{2<  �          B(��
=�UG����u��Cl(��
=?޸R�{z�C(��                                    Bx{@�  �          B�R�
=�R{��{�tG�CjE�
=?���� 
=�\C)�                                    Bx{O�  �          B\)�\)�N�R����r�
ChaH�\)?�z���{��C)!H                                    Bx{^.  �          B��\)�Up���R�r��Cj���\)?��
����C*��                                    Bx{l�  
�          B��{�^{��(��q�Cmc��{?�G�� ���C-�                                    Bx{{z  �          Bff���v=q��
=�l�
Ctk������  ffC8
                                    Bx{�   �          B  ���
��G������fffCy{���
��z��p���CD�q                                    Bx{��  
�          B���z���{��{�e\)Cx{��z����� ��CDY�                                    Bx{�l  �          B�H���H����ڏ\�_�
C~  ���H�P������HCT��                                    Bx{�  
�          B�������\��z��]G�C~�=����l(��Q�z�CX�H                                    Bx{ĸ  �          B����(���\)��\)�X��C}  ��(���ff� �\W
CX��                                    Bx{�^  �          B�R��G������(��i33Cy�\��G��У��Q��CC�                                    Bx{�  �          B�����������`C|� ���E�G��CQ��                                    Bx{�  �          B�\���
��  ��
=�h  C{33���
��(��\)#�CGQ�                                    Bx{�P  T          B���(���ff�ݙ��f  C|@ ��(���=qk�CK��                                    Bx{�  �          B=q��\)�������_  C{
��\)�Mp�� ff�CP@                                     Bx{�  T          B���\)��p���Q��]33C�3��\)�y���ff33C]�                                    Bx{+B  �          B���{�~�R�߅�i�Cw5���{���R� z��C@�\                                    Bx{9�  �          BG�����k\)��\)�i
=Cn\)��þ�(���ff�C6�q                                    Bx{H�  �          BG��*�H�Vff����j{Cgs3�*�H?@  ��(��C0�                                    Bx{W4  T          B=q�0Q��N�R��Q��k33Ce���0Q�?�Q�����u�C-�)                                    Bx{e�  
�          Bff�=p��G33��(��i��Cbs3�=p�?�=q����ffC,h�                                    Bx{t�  
(          B��733�Qp���33�hQ�Cd�{�733?p����z��C/G�                                    Bx{�&  "          B�R�.�H�W�
���h��Cf���.�H?����ff#�C0�H                                    Bx{��  T          B�
�0  �]p���ff�f33Cg���0  >L����z���C2�R                                    Bx{�r  �          B
=�'�
�j�R��G��c�\Cjp��'�
�!G���ff�fC7k�                                    Bx{�  
�          B�H��x(����H�`ffCm������
��G��qC<��                                    Bx{��  �          B���p��xz���Q��`�Cm�H�p��˅���H�RC=&f                                    Bx{�d  �          B�\�����=q����_  Cp�������R��=q(�CA                                    Bx{�
  �          B��������{�b�RCtxR����\)� ��� CC�                                    Bx{�  �          B���33���������`Cs���33��H� z�.CDs3                                    Bx{�V  �          BG���ff�����܏\�`�Cw33��ff�<���=q�CJE                                    Bx{�  "          BG������=q��p��j
=Cv����녿����ffǮCA=q                                    Bx{�  �          BG��#��f�H��ff�g{Cj���#���Q���ff{C6�                                    Bx{$H  T          B��8Q��Z=q��\)�f�Ce���8Q�>Ǯ��Q��\C2\                                    Bx{2�  �          B�$  �i���(��f�HCj���$  �����\=qC6�
                                    Bx{A�  �          Bp��2=q�S
=��=q�kG�Ce�
�2=q?h����\)�
C/O\                                    Bx{P:  �          B�\�%��l�����
�d�RCk#��%��J=q��33��C8^�                                    Bx{^�  �          B�H�G����\����[33Co���G��0����p�
=CD�                                    Bx{m�  �          B���ff��z���Q��d33Cs��ff��� ���CB�                                    Bx{|,  B          BQ��{��(����H�bz�Css3�{�z���\)CC��                                    Bx{��  4          B{�{������H�_�
CqJ=�{��������CC�                                    Bx{�x  �          BG���
���R�ָR�Y
=CrB���
�[���
=.CH޸                                    Bx{�  !          B{�����\��z��Y{Cs�����j=q����CK\)                                    Bx{��  T          B�
��(���=q���
�Y(�Cu�\��(��y�����

=CNW
                                    Bx{�j  
�          B�
�=q��ff�ҸR�TG�Cu���=q���������CQ=q                                    Bx{�  �          B������\�����R
=Cv��������=q� CS(�                                    Bx{�  T          B���(���
=��\)�R�
Cu���(�������
W
CQ�
                                    Bx{�\  �          B���R��33���H�Q�RCx0���R�����G��)CW
                                    Bx{   T          B33������R����OQ�Cy&f������R���
�CZ#�                                    Bx{�  
�          B����(��������H�MG�C|
=��(��ʏ\����C`�q                                    Bx{N  "          B�������=q���H�JffCx8R������������qCZ��                                    Bx{+�  �          B�H���H��p���G��G�Cz�����H�߮��p��C`��                                    Bx{:�  
�          B�H��
=���
��ff�K��Cxc���
=�ƸR��L�CZ�
                                    Bx{I@  
(          B�H��\��p������I�RCv�
��\��Q���Q�8RCX��                                    Bx{W�  �          B�R��=q��  ��{�A
=Cy����=q���H���R��Cb                                    Bx{f�  
(          B���ə������  �<�C}���ə���H��=qz�Cj�=                                    Bx{u2            B�����������H�@��C~5�����������qCjQ�                                    Bx{��  �          Bff�����\��(��]��CwxR���mp�� ff�RCOT{                                    Bx{�~  e          Bz����
��z����H�S�Cy����
��
=��\)=qCZ�                                     Bx{�$  T          BQ���  ��=q��{�L��CzJ=��  ��z�����3C^k�                                    Bx{��  �          Bff��Q������{�O(�Cy8R��Q����R���C[\)                                    Bx{�p  T          Bp����H��(�����NCy�����H��33��(�33C\�3                                    Bx{�  �          B=q�����z���G��N=qCy������{��
=C]B�                                    Bx{ۼ  
�          B33��=q������ff�P  Cy����=q��{����\C\�                                    Bx{�b  "          BG���G���=q�����J�Cx�H��G��љ���=q�
C\�R                                    Bx{�  T          B=q��������(��L�
Cx������=q���R�C\aH                                    Bx{�  �          B=q������
��\)�T(�Czn�����Q���G��C[�H                                    Bx{T  �          BG��Ϯ��\)�ҏ\�S{C{��Ϯ��  ��33��C]��                                    Bx{$�  
�          B{��G���p��Ώ\�M��Cz�q��G���{��\){C_xR                                    Bx{3�  T          B�H��ff���\��33�I��Cz����ff�߮����33Ca0�                                    Bx{BF  �          B�H����Q���\)�G{CzJ=���������ǮCa�                                    Bx{P�  �          B�
��  ����Ǚ��D�RC{�q��  ������z��Ce#�                                    Bx{_�  �          B����
=��p���  �B��C{�q��
=������qCfQ�                                    Bx{n8  �          B���H����ǅ�D�Cz����H������=Cb�                                    Bx{|�  �          B����{������D
=Cy.��{�������\C`�f                                    Bx{��  �          B��љ���33���A��C|�
�љ�����(�p�Cg�                                    Bx{�*  �          B���Q���z���Q��=\)C~n��Q���\��G�=qCl                                    Bx{��  �          B
=��{���\����NC}���{��ff��  � CdxR                                    Bx{�v  �          B���߮���\��{�S�RCyaH�߮�����G�L�C[�                                    Bx{�  T          B��
=�������K\)Cy�f��
=������R�)C_Y�                                    Bx{��  T          B���������\��{�VCx�������z���\)�{CW=q                                    Bx{�h  T          Bff���
��(���Q��T��Cx����
����ff��CXaH                                    Bx{�  "          BQ���\)��(���33�SG�Cw���\)��  ��\)ffCWxR                                    Bx{ �  �          B33��=q����У��R�Cw����=q�����G�z�CX�=                                    Bx{Z  �          BG���ff��p��У��RCw�q��ff��
=��\)�RCY�                                     Bx{   �          B33��=q��=q�ҏ\�U�\Cw=q��=q������Q�CV��                                    Bx{,�  �          B�\��ff��Q���ff�T\)Cw�
��ff��G���ff33CX�f                                    Bx{;L  
�          B�������=q��ff�T=qCw�=�����G���ff�fCXE                                    Bx{I�  �          B�����H��p���p��UCx���H��p�������CXs3                                    Bx{X�  
�          B�
�陚��(����H�W\)Cw�3�陚��
=���aHCW�{                                    Bx{g>  T          B�����R�����(��Vp�Cv����R��
=���H(�CV�                                    Bx{u�  T          B�����������33�X(�CvG�������ff����z�CT}q                                    Bx{��  
�          B��R���������Z�Cw�q��R�����(�u�CU�3                                    Bx{�0  �          B���\)��z����
�X��Cw5���\)������8RCV
=                                    Bx{��  �          B��G���G��Յ�X�Cv33��G���ff���Hp�CTxR                                    Bx{�|  �          B�H��=q��=q��Q��_p�Cu���=q�z�H��Q�=qCN��                                    Bx{�"  �          B��	G��33��ff�bCq�R�	G��E���p��fCG�q                                    Bx{��  �          B=q��\)�������T�
Cv8R��\)���������)CV��                                    Bx{�n  �          Bff�ٙ������=q�@Q�C|  �ٙ���R��  8RCh��                                    Bx{�  �          B����\)��33��G��9�C~����\)�!�����ffCoG�                                    Bx{��            B=q��z����\��=q�DffC}aH��z��(���z���Cj33                                    Bx{`  3          B��������H����>C}^�����(���{��Ck                                    Bx{  T          A�p��n{�����{�>�C�f�n{�=q��33(�Cx��                                    Bx{%�  T          A����G���������.{C~33��G��4(���33G�Cp�=                                    Bx{4R  �          B G���(���p�����/��C}�R��(��1������
Cp+�                                    Bx{B�  
Z          B������������#��C�4{�����S\)��R.C|��                                    Bx{Q�  �          B�Tz����
��33� �C�"��Tz��\����RC�9�                                    Bx{`D  "          Bff��{������R�'ffC�y���{�I��ffCv��                                    Bx{n�  
�          B���N{�¸R�����#�C�<)�N{�XQ���G�ffC�N                                    Bx{}�  "          B�H�'
=�������
�(�C��=�'
=�Q���33�C�\)                                    Bx{�6  �          A��
@������+�
��p�C�<)@����
��G��0ffC��                                    Bx{��  
�          A���@�p���\��{�+�
C��R@�p��Ů����	\)C��                                     Bx{��  
�          BQ���R��ff�����C�<)��R�n{���{�C�Ff                                    Bx{�(  
�          B�� ���ʣ���{�z�C�*=� ���h���홚�~Q�C�R                                    Bx{��  "          B�R��  ��{����#�C�h���  �]����p�C}��                                    Bx{�t  �          B��������
=����#��C��\�����\  ��R\)C|@                                     Bx{�  �          B�����Ǚ���\)� ��C�޸�����b�H��\)�}ffC|W
                                    Bx{��  
�          B\)���H�����������C��R���H�m��G��w33C{�3                                    Bx{f  
�          B��g����
��ff�%�C��=�g��Z=q��z�Q�C!H                                    Bx{  
�          B\)�I��������,�C�9��I���K
=����{C�
=                                    Bx{�  
�          Bp��L���������R�+ffC�0��L���N�\��Q�=qC��                                    Bx{-X  T          B=q�<(���33��  �(�C����<(��U�����)C��f                                    Bx{;�  �          B�Dz������ff�%  C�h��Dz��[�
��{  C���                                    Bx{J�  �          Bff�Q���p���  �(33C�#��Q��V{���k�C��                                    Bx{YJ  T          B��hQ��Ņ��G��"�C��3�hQ��a���
�~��C�
                                    Bx{g�  "          Bp��XQ���Q���z��!�RC���XQ��d(���\)�~�C�U�                                    Bx{v�  �          B�
�vff�������$(�C����vff�]G������HC~n                                    Bx{�<  T          B{��
=���\��33�)Q�C�U���
=�O33�뙚�
Cy�3                                    Bx{��  
�          B
=�����������0\)C��q����A�����HaHCz+�                                    Bx{��  �          B�H�N�R��  ����.�
C���N�R�G���ff=qC}q                                    Bx{�.  �          B���
=���H����,z�C�q��
=�N�R���p�C�\)                                    Bx{��  
�          B\)@mp���ff��z��z�C���@mp�������Q��`�C��H                                    Bx{�z  
(          B����
=��  �����C��쿗
=������H�q(�C��                                    Bx{�   �          B���Q��ʸR��ff��C�3�Q��s����H�x��C�u�                                    Bx{��  
�          Bff?����������z�C�H�?���t����(��w��C�                                      Bx{�l  �          B��?�33������ffC�L�?�33�uG���=q�w�\C�&f                                    Bx{	  
�          B�R?��R���������Q�C���?��R���H���
�o=qC���                                    Bx{�  
�          B�H@
�H������z��
33C�O\@
�H���
�ܣ��eC���                                    Bx{&^            Bff@1G���ff��33�{C�
=@1G���\)��=q�m(�C�˅                                    Bx{5  	�          B�H@W���������\C�Ǯ@W��y���  �r  C�
                                    Bx{C�  �          B33@0����p����
��RC���@0�����\��=q�g�C��
                                    Bx{RP  
�          B=q@B�\�ҸR��
=���C�K�@B�\�����
=�hG�C�
                                    Bx{`�  �          B(�@>�R��
=���\�\)C�9�@>�R��(��܏\�g�RC��R                                    Bx{o�  
�          B�@P  ��p���G��{C���@P  ��33�ۅ�f�C�`                                     Bx{~B  3          B�@QG�����=q�(�C��f@QG���  �ڏ\�e{C�`                                     Bx{��  T          B�
@e��p������C���@e������=q�az�C���                                    Bx{��  "          B�
@@����p����
��RC�7
@@�����������bz�C�ٚ                                    Bx{�4  
�          B�
@"�\��z����\���C��H@"�\��ff����g��C�<)                                    Bx{��  �          B33@W
=��  �����G�C��)@W
=���R�ڸR�d�\C�z�                                    Bx{ǀ  
�          Bp�@Q���ff��\)�
=C�P�@Q������Q��iC���                                    Bx{�&  3          B��@���х������C�e@����=q��p��k(�C���                                    Bx{��  �          Bff@Q�������G��33C�T{@Q������G��k��C��H                                    Bx{�r  
          B�?�
=������
=�(�C�N?�
=��z�����o�RC��                                    Bx{  �          B�\?��
���
��p��\)C��?��
��=q��R�m�C���                                    Bx{�  
�          B=q?��������{�\)C��H?���������p��lz�C��q                                    Bx{d  T          BG�?������
=�33C�q�?��������{�m=qC�C�                                    Bx{.
  T          B  ?�G��ͮ��z��=qC��\?�G���Q����\�o{C��f                                    Bx{<�  �          B�
?���ͮ�����
C���?��������  �nffC�!H                                    Bx{KV  �          B�
@�������ffC�U�@�������z��k�RC��H                                    Bx{Y�  "          B��?��ͅ���
��
C��?������߮�n{C�5�                                    Bx{h�  �          B�R?�{��(���ff�
=C��f?�{���R��\)�j33C��q                                    Bx{wH  �          Bff?�
=��Q��������C��?�
=�~{��
=�q��C��                                    Bx{��  	�          B=q?�  �ʸR���
���C���?�  ��
��=q�p��C��                                    Bx{��  3          B�?����х�����C�?�����\)��G��g(�C�{                                    Bx{�:  �          B�R@��Ӆ����ffC�Z�@���ff��\)�cC��=                                    Bx{��  	�          B�
@!��ң���z���C�� @!���G���=q�d�RC�#�                                    Bx{��            B�H@*=q��Q������{C��@*=q������ff�d�C�Y�                                    Bx{�,  
Z          B@�R��(�����z�C�n@�R���H��z��e\)C��=                                    Bx{��  
�          Bp�@��х���H���C�~�@����\����eQ�C�                                    Bx{�x  �          B\)@��ң��������C���@���ff��ff�c(�C��
                                    Bx{�  
�          BG�@(���(���33�{C��f@(���  ��ff�cffC���                                    Bx{	�  T          Bff@_\)��Q���z���\C��{@_\)���\�Ӆ�[=qC�c�                                    Bx{j  �          B�\@1���  ��(���C�@1����
��
=�c�C���                                    Bx{'  
{          B�\@#�
��
=���ffC�˅@#�
��z���{�e33C�33                                    Bx{5�  �          B��@!���z������\)C��f@!����ڏ\�f  C�.                                    Bx{D\  �          B�\@:=q������{��C�4{@:=q��{��33�g33C�ٚ                                    Bx{S  T          B�\@1G���=q��G��G�C��@1G���
=��  �hp�C���                                    Bx{a�  T          B�@.{�ͮ��z��p�C�@.{��=q�����iz�C���                                    Bx{pN  �          B@
=��
=��Q��Q�C�Z�@
=�����=q�kz�C��q                                    Bx{~�  T          B{@Dz���=q���C�u�@Dz������Q��n=qC�]q                                    Bx{��  T          B��@Y����{�����{C�޸@Y���}����p(�C��                                    Bx{�@  "          B��@G�������{�(�C���@G��{�
����q=qC���                                    Bx{��  �          B\)@vff���
��=q��C�w
@vff�t�����
�sG�C�\                                    Bx{��  
�          B ��?����z���ff���C���?����{���jffC�                                    Bx{�2  T          B G�?�{�Ə\���
�ffC�&f?�{�}p��ޏ\�p�C�Z�                                    Bx{��  �          B =q@���������C���@���v�R��{�s�C�(�                                    Bx{�~  T          B z�@{�Ņ��G���HC���@{�{33��p��q��C��                                    Bx{�$  "          B z�@
=q��p������33C�xR@
=q��ff���
�n�HC��3                                    Bx{�  
�          B z�?��
�Ӆ����	�\C���?��
�����G��]G�C�˅                                    Bx{p  �          B \)?�z���������
=C���?�z���  ��p��Z�RC�.                                    Bx{   
�          B ff?��
��Q�����\)C��?��
������=q�[�
C�`                                     Bx{.�  
�          B ff?�\)��{�����RC�u�?�\)������z��Y{C�
                                    Bx{=b  
�          B p�@_\)�����ff�!z�C��@_\)�s���ff�s�C�u�                                    Bx{L  �          B ��@�33��z����%(�C��H@�33�k���Q��vp�C��                                    Bx{Z�  
�          B �@qG���G�����'ffC��f@qG��hQ��㙚�x��C�Ff                                    Bx{iT            B ff@N�R��������#  C���@N�R�r{���H�t�C��                                    Bx{w�  �          B �R@��������\���C��3@��|z��߅�q
=C�b�                                    Bx{��  �          B �?�\)�˙���ff�  C���?�\)��p���(��hQ�C�Q�                                    Bx{�F  T          A�?�Q����������33C��?�Q������ff�gG�C�ٚ                                    Bx{��  "          A�p�@+��Ǯ�����z�C��@+�����ٮ�j  C��H                                    Bx{��  T          B �?W
=�ϙ�����G�C��?W
=�����33�aG�C�Z�                                    Bx{�8  T          B ff?n{��z�������C�f?n{��������`p�C�~�                                    Bx{��  
�          B �\?�ff��Q������RC�j=?�ff����G��]Q�C�
=                                    Bx{ބ  "          B ��#�
���
��Q��p�C��)�#�
��33��ff�e  C���                                    Bx{�*  "          B  ?Q���  ��  ��C�� ?Q����\��p��Y
=C�>�                                    Bx{��  T          B ��>�ff��=q��z��
ffC�z�>�ff��=q��
=�[��C���                                    Bx{
v  T          B ��>�(���ff����	\)C�xR>�(��������Zp�C���                                    Bx{  "          B �?��������Q���C��?���������YffC��=                                    Bx{'�  
�          B �R?����ң������HC�N?�����������\�C�޸                                    Bx{6h  T          B ��?�ff��
=���R�{C�h�?�ff������z��[�\C��                                    Bx{E  �          B p�?����z����H��C�u�?�������Q��[��C�
                                    Bx{S�  
Z          B =q?����33����RC���?�����ҏ\�\C��3                                    Bx{bZ  
{          B �?�\)�х��33�G�C�9�?�\)��z���{�\G�C��                                     Bx{q   �          A��
?�  ��
=��
=�Q�C�^�?�  ��=q�љ��\(�C���                                    Bx{�  T          A��?�p���G���=q��C�Y�?�p����H���H�[Q�C��                                    Bx{�L  
�          A���?�{��
=���
�	  C�4{?�{��p�����X�C��3                                    Bx{��  T          A��?s33�ԸR��  ��
C�f?s33�����
�V=qC�n                                    Bx{��  �          B {?��\��G���=q��C�  ?��\��  �ә��^�HC��q                                    Bx{�>  �          B �?�
=��z���z���C��3?�
=�����=q�\��C�>�                                    Bx{��  �          B (�?����z������
=C�4{?�������ff�_�C��q                                    Bx{׊  �          B G�?k�������\)���C�f?k����
��Q��_\)C�xR                                    Bx{�0  �          B Q�?J=q�υ���H�
=C�޸?J=q��������^��C�@                                     Bx{��  T          B ff>�ff�Ϯ��
=�{C�}q>�ff���H��  �^�\C��{                                    Bx{|  T          B �\>������H���
���C�U�>������\��33�\�C�y�                                    Bx{"  �          B ��?z����������C��H?z�������33�Y�C���                                    Bx{ �  T          B �\?\(��ә�����
Q�C��\?\(����R��(��X=qC�P�                                    Bx{/n  �          B ��?��������
���C���?����H�Ώ\�U��C�!H                                    Bx{>  "          B ?ٙ��֏\���
��C�Ф?ٙ���33����S�C��H                                    Bx{L�  
�          B ��@	����p���(���
C�G�@	�����R�ˮ�Q
=C�                                      Bx{[`  T          B �H@����
=�����Q�C���@����Q�����Q=qC��3                                    Bx{j  �          B �
@�H���������z�C���@�H��Q�����Q=qC��                                    Bx{x�  T          B ��@�
��z���p��33C�u�@�
���
��=q�Q�
C�aH                                    Bx{�R  
�          B @�
��  ��Q��33C�33@�
��33���H�RC��                                    Bx{��  T          B �\?�z��ԏ\��{�Q�C���?�z���p����
�T��C�xR                                    Bx{��  
�          B ��?�\)��Q��������C��)?�\)��{��  �Q��C�aH                                    Bx{�D  �          B p�?�p��֏\������C��3?�p����R��33�Q�C�'�                                    Bx{��  �          B �?����ָR������RC��{?�����
=��
=�PC�S3                                    Bx{А  
Z          B �?���ׅ��=q�
=C���?����{�����O�C�:�                                    Bx{�6  	�          B �H?��ٮ��G�� C��H?���G��ȣ��Lz�C�]q                                    Bx{��  �          B �
?���{���H� Q�C�}q?����
��=q�K�C�H                                    Bx{��  �          B �?޸R����������ffC��3?޸R��p���=q�I��C�p�                                    Bx{(  �          B ��@Fff��G��{���{C�+�@Fff������G��?�HC�"�                                    Bx{�  
�          B  @z�H��{�pz���Q�C��{@z�H��33���R�9z�C�{                                    Bx{(t  	�          BQ�@3�
��G���33��(�C���@3�
��{�Ə\�H�C��f                                    Bx{7  �          B��@E��\�o���=qC�
@E��  ���H�8z�C��{                                    Bx{E�  �          Bp�@�(���{�b�R���C�
@�(�������
=�1��C�(�                                    Bx{Tf  
�          B��@�ff��
=�Q�����RC���@�ff��\)��33�'��C�\                                    Bx{c  	�          B�HA)��=q�����C��A)���H�����{C��f                                    Bx{q�            B��A.=q��� (���p�C�O\A.=q��  ������C�
                                    Bx{�X  T          B�
A5���\)�{��Q�C���A5��Å�����p�C�y�                                    Bx{��  
Z          Bz�A3�����=q���HC�� A3���\)���\���C�Y�                                    Bx{��  �          B�A4������  ����C���A4�����
��G��p�C�aH                                    Bx{�J  T          B�\A6=q���{�~�HC��)A6=q��p�������\C�c�                                    Bx{��  
(          BA6�\�������Q�C���A6�\������(���
C�l�                                    Bx{ɖ  "          Bz�A#33��{�+����HC�A#33��p�����
C��=                                    Bx{�<  �          B  A
=��G��<  ����C��=A
=��(������RC�g�                                    Bx{��  �          B ��A�H����>=q����C��A�H���\��Q���C���                                    Bx{��  �          B p�@�����B�\��{C�O\@���  ��=q�{C��R                                    Bx{.  T          B =q@�=q��p��e�����C�Y�@�=q��Q�����1��C��
                                    Bx{�  
�          B �@�Q����TQ����C��@�Q������Q��(\)C���                                    Bx{!z  
�          BG�A&=q��ff�"ff���RC��=A&=q�����R�C���                                    Bx{0   �          B{A{����'���Q�C���A{����\)�G�C��
                                    Bx{>�  T          Bp�A  ��(��2{��\)C�c�A  ��G���z��(�C���                                    Bx{Ml  
�          B�A���33�)�����C�S3A��î�����33C��\                                    Bx{\  
�          B ��A���{�<����C�7
A����
���\�(�C���                                    Bx{j�  �          B  @�33��p��G33��C���@�33����G��p�C�>�                                    Bx{y^  T          B �\A-G����
=����C�W
A-G���G���=q���C��
                                    Bx{�  �          B �Aff��p��.�R���C�ٚAff����\)�{C�s3                                    Bx{��  
�          B   A��癚�%����\C�xRA���\)��G����C���                                    Bx{�P  
�          B   A{��\)�   ����C��)A{��  �������C�y�                                    Bx{��            A�
=A=q��(������RC�0�A=q��\)���H�{C��3                                    Bx{  e          A�z�A=q��p��&�\����C��)A=q��G�����  C�8R                                    Bx{�B  
�          A�(�AG�����  ��\)C��
AG�������=q�=qC�b�                                    Bx{��  �          A�ffA8Q���������yC�\A8Q��Ù�������
C��q                                    Bx{�  
(          A���A/�
�㙚�{��\)C��\A/�
��
=��=q�z�C�!H                                    Bx{�4  T          A�33A"�\��\)�(����HC�A"�\���
�����
C�E                                    Bx{�  �          A���A  ��Q��(���ffC�Y�A  ��=q�����
C���                                    Bx{�  T          B {A8  ������pz�C��A8  �����{��ffC�e                                    Bx{)&  "          B   A#\)��������\)C���A#\)�¸R���\�	�\C�aH                                    Bx{7�  
�          A��A2ff��ff�����
C��=A2ff��ff��\)���C�5�                                    Bx{Fr  T          B �A0  �噚��
��ffC�|)A0  ���
�����=qC���                                    Bx{U  
�          B   A33��ff�'����HC��)A33�����ff�C�T{                                    Bx{c�  �          B 
=A�����H�+33��=qC�:�A����G������C���                                    Bx{rd  �          B �A�����4z���
=C�\A���
=�����Q�C���                                    Bx{�
  T          B   A z���\)�+33��z�C��
A z���  ���H�Q�C�U�                                    Bx{��  "          B {A-������� Q���  C�u�A-����\)��\)�G�C��                                    Bx{�V  
�          A��A?�
���
������C���A?�
�£�������
C��                                    Bx{��  "          B (�AG�
���\������C���AG�
��p���33����C���                                    Bx{��  �          B 33AC\)��33�(���ffC��RAC\)�������� �HC�aH                                    Bx{�H  "          A��
AD������(���z�C��3AD����G���Q�� ��C�|)                                    Bx{��  "          A�G�AC
=���������C�ФAC
=���\��G��33C���                                    Bx{�  	�          A��HA4����=q�5G����RC�%A4����z������
=C��                                    Bx{�:  �          A���A0  �����?\)���RC��3A0  ��������
=C��=                                    Bx{�  �          B =qA�  ���H�i�����C��=A�  ��33�5G���(�C���                                    Bx{�  �          B Q�An{���
��\)�,Q�C�)An{�Ə\�[�����C�xR                                    Bx{",  �          B G�Ajff��G���\)�IG�C��3Ajff���
�jff��\)C�q�                                    Bx{0�  �          B 33Ad  �ޣ�����JffC��\Ad  ����k\)�ڣ�C�                                    Bx{?x  
�          B �AR{�������g\)C���AR{���H�{33����C�
=                                    Bx{N  �          A��
A^ff�ڣ������
C�y�A^ff��p����
���RC�4{                                    Bx{\�  
�          A��AR�\�י��,(����C�AR�\��{��  �	��C�f                                    Bx{kj  �          A�=qA]p�����#33��C��\A]p����
��
=���C���                                    Bx{z  �          A�\)A[\)���+\)��{C���A[\)���R��=q���C��                                    Bx{��  T          A���AO������)p���ffC���AO���  ��{���C��R                                    Bx{�\  
�          A�  AI������3
=��z�C���AI���
=��{��RC��=                                    Bx{�  �          A�(�AM��p��5���C���AM��33���H��C��                                    Bx{��  
�          A��\AI���=q�0(���33C��3AI�����������C��=                                    Bx{�N  
          A�G�AM����H�+\)��Q�C�˅AM���(����\�	{C���                                    Bx{��  �          A�\)AG������)G���{C�XRAG����\��  �\)C�1�                                    Bx{��  T          A���AS\)�ظR����ffC�HAS\)��{��  ��RC���                                    Bx{�@  �          A�33A]G���ff�{��Q�C���A]G���  ���\��{C�<)                                    Bx{��  �          A�\)AR{��Q��\)��p�C��RAR{��33�����ffC���                                    Bx{�  �          A��AO���=q�  ����C��RAO����\������\C�j=                                    Bx{2  
�          A�z�A:�\��{�����bffC��A:�\����u��\)C�c�                                    Bx{)�  
�          A�33AZ�H�ᙚ�����.�RC��AZ�H��p��Y����C�#�                                    Bx{8~  
(          A�33AP��������V�\C�k�AP����z��n=q��Q�C��H                                    Bx{G$  �          A��HA5���G���=q�MG�C���A5������l  ����C��3                                    Bx{U�  "          A�ffA&�R��z���
=�<(�C���A&�R��G��dQ��ՅC���                                    Bx{dp  "          A��A`Q��ڸR�=q�tQ�C��3A`Q���  �x�����C�&f                                    Bx{s  "          A�
=A[33�ۙ��	���z�\C�B�A[33��z��|z���\)C��{                                    Bx{��  �          A�
=AY�������R�n{C�
AY��£��v{��\)C��3                                    Bx{�b  �          A���AT(��ޣ���\�tQ�C��qAT(�����z�\���HC�8R                                    Bx{�  �          B 
=AX����ff����pQ�C�HAX����  �xz���{C�z�                                    Bx{��  "          B {AV�H���H�����G�C�3AV�H��\)������\C��=                                    Bx{�T  
�          B 
=AE��߅���=qC���AE�������z����C�t{                                    Bx{��  T          B {A���ff��  �;33C�P�A��م�e����
=C�"�                                    Bx{٠  �          B �RA�\��G���z��\)C�s3A�\��33�S33��  C�{                                    Bx{�F  
�          B33A#�
������p��9�C���A#�
��z��b{�ѮC�s3                                    Bx{��  �          B �
AG�
���_33��G�C���AG�
��=q�����C�                                    Bx{ �  T          B �\AQ��ԏ\�F�\���C�RAQ���Q���ff���C�0�                                    Bx{ 8  �          B AQ��؏\�7�����C���AQ���=q��(��
G�C��{                                    Bx{ "�  T          B ��AW\)�Յ�<����z�C�aHAW\)���R���
�ffC�e                                    Bx{ 1�  �          B �AF{�ܣ��/33��G�C�
AF{��p������{C���                                    Bx{ @*  
�          B ��A$Q���Q���G��B�RC���A$Q���p��g
=����C�o\                                    Bx{ N�  
�          B ��A\)��  ��z��T  C�AHA\)��  �o�
�ݙ�C�5�                                    Bx{ ]v  T          B �A (���z�����M�C�G�A (������l(����
C�7
                                    Bx{ l  �          B   @������7�����C��@������.ff��\)C�S3                                    Bx{ z�  
�          B \)A=q��ff�q��أ�C�g�A=q���
�;����C��=                                    Bx{ �h  T          B Q�A ����Q����H��=qC�S3A ������@  ��C��R                                    Bx{ �  �          B Q�A33��
=������z�C��
A33��\)�D  ���C�G�                                    Bx{ ��  "          A�\)@��������s�
�ۅC�!H@�����33�:�R���C���                                    Bx{ �Z  �          A��H@����
�s33�ۅC�ٚ@����:�\��C�Q�                                    Bx{ �   T          A���@������33�  C�K�@����Ip���{C��)                                    Bx{ Ҧ  T          A���A����z��Ǯ�4��C��3A����\)�\����\)C���                                    Bx{ �L  �          A���AG���  �׮�C�C�}qAG���  �c33���
C�K�                                    Bx{ ��  �          A�z�Az��홚��  �K33C�u�Az�����f�\��\)C�H�                                    Bx{ ��  "          A�=qA�����=q�F=qC�L�A����c����HC�
                                    Bx{!>  
�          A�=qA�\���
��33�G33C�W
A�\���
�c�
��
=C�"�                                    Bx{!�  �          A��A
�\�����33�@��C�  A
�\��=q�_���C��H                                    Bx{!*�  �          A��HA\)��������RC�nA\)�����  ���C���                                    Bx{!90  "          A�
=A��噚��R���C�}qA����
�
=��  C��
                                    Bx{!G�  
�          A�
=A��癚�
�R�~�RC���A���Q��{�
����C��\                                    Bx{!V|  T          A��A(����z��ff��=qC�7
A(����{��(����C�o\                                    Bx{!e"  �          A�
=A=q��33�  �yG�C�9�A=q��z��xz���33C�=q                                    Bx{!s�  �          A��A�
��R��
=�aC���A�
�ә��m���=qC�b�                                    Bx{!�n  �          A��A����������m�C�  A������s���{C��q                                    Bx{!�  �          A�33A���33�
=�w\)C���A��иR�w�
��ffC�|)                                    Bx{!��  d          A�
=A�R�������=qC��A�R��=q�{
=��  C��                                    Bx{!�`  "          A�33A=q��(��  ��(�C�B�A=q�����~�\��\)C�P�                                    Bx{!�  "          A�33A z�����33���C���A z���
=��=q��C�Ǯ                                    Bx{!ˬ  �          A�G�A
=��G���
�x��C�B�A
=����v�R��33C�AH                                    Bx{!�R  T          A��A{��G��
�H�~=qC�5�A{�����yG���C�7
                                    Bx{!��  T          A���A33��(������33C���A33�υ�{\)��ffC��                                    Bx{!��  �          A���A�������{�C��)A��љ��xQ���p�C���                                    Bx{"D  "          A�z�A��������(�C��=A���=q�~{��C��=                                    Bx{"�  T          A��\A33����{�v=qC��A33��=q�t����(�C��=                                    Bx{"#�  �          A�  A�����
�
ff�33C�u�A�����x  ���
C�`                                     Bx{"26  �          A���A	��������z�C�O\A	��Q��y����RC�9�                                    Bx{"@�  "          A�(�A\)��(��{�qp�C��A\)��33�o���33C���                                    Bx{"O�  T          A���@��R���Q���=qC�� @��R�̣���
���\C��                                    Bx{"^(  "          A��\@�ff��\)�?
=����C��H@�ff��p������(�C�xR                                    Bx{"l�  
�          A�ff@�{���H�x  ��(�C���@�{���\��
=�.z�C��                                    Bx{"{t  �          A�Q�@�=q��\)�B�\���C�{@�=q��
=�����33C��)                                    Bx{"�  �          A���@�Q��߅�K33��\)C�O\@�Q���Q������{C�J=                                    Bx{"��  T          A��H@�=q���H�Jff��ffC��H@�=q�����  �33C��                                    Bx{"�f  T          A�p�@�p�����@����(�C�%@�p���\)�����
C�.                                    Bx{"�  �          A�@׮��R�>�R��  C��\@׮��33������
C���                                    Bx{"Ĳ  "          A�  @�ff���I���{C�33@�ff��
=����  C�#�                                    Bx{"�X  
�          A�z�@�
=��ff�=q��Q�C��
@�
=�ә���\)���\C�\)                                    Bx{"��  T          A��R@��\�����\����C��3@��\���
�������C�5�                                    Bx{"�  �          A��@�
=���H�{��p�C�7
@�
=��\)�������RC���                                    Bx{"�J  �          A�G�@Ϯ���&ff���C�E@Ϯ��
=��Q��33C�
=                                    Bx{#�  
�          A�G�@Ӆ��
=�'���z�C�c�@Ӆ�Σ����\��C�,�                                    Bx{#�  
�          A�33@�\)���
�!p����RC�|)@�\)��(��������C�AH                                    Bx{#+<  
�          A��R@߮��
=�+
=��Q�C���@߮��ff�����  C��f                                    Bx{#9�  "          A���@�z������,Q����C��{@�z���=q�����  C��=                                    Bx{#H�  
�          A�
=A�����
�+�
���HC�Y�A����G������33C�`                                     Bx{#W.  T          A��A����=q�.�\��p�C��HA���ǅ���
�
=C���                                    Bx{#e�  �          A�33A33���/
=���
C�C�A33������Q���C�J=                                    Bx{#tz  
Z          A��@�(���z��2�H��G�C��
@�(���\)��=q�p�C���                                    Bx{#�   �          A�G�@����Q��3\)���C��H@�����������C��q                                    Bx{#��  
Z          A��A���z�������C�<)A�����s�
��z�C�(�                                    Bx{#�l  T          A�(�A,���������w�C�U�A,����
=�m��߅C�Q�                                    Bx{#�  
�          A�  A
�R��z��33��ffC�^�A
�R���
���R��Q�C�O\                                    Bx{#��  
�          A�(�AH���ۅ������C�H�AH����\)�~�H��
=C��
                                    Bx{#�^  �          A��A/
=��=q�'����\C�� A/
=������������C���                                    Bx{#�  T          A��RA���(��3
=��  C�=qA��ř������
=C�b�                                    Bx{#�  "          A�G�A�\��p��2�\��
=C�HA�\��
=���\���C�)                                    Bx{#�P  �          A��A�H��
=�.�\���C��)A�H��
=���H��RC�Ǯ                                    Bx{$�  �          A���A���p��,  ����C��HA����
����\)C��=                                    Bx{$�  "          A��A�����,  ���RC�@ A���
=���
���C�9�                                    Bx{$$B  �          A��A����33�(  ����C�K�A����(����
����C�>�                                    Bx{$2�  �          A�p�@����R�&ff��\)C���@�������G���  C���                                    Bx{$A�  
�          A��@Ϯ��  �!���G�C�B�@Ϯ�Ѯ����
=C���                                    Bx{$P4  "          A��@��癚�)���{C��
@��Ώ\����� �C��f                                    Bx{$^�  �          A�p�@����0Q����HC�l�@��������\C�P�                                    Bx{$m�  
�          A�G�@أ���  �+
=���C��3@أ����H��p��ffC�W
                                    Bx{$|&  "          A���@�=q��z�������C�U�@�=q�ԣ����R��Q�C��                                    Bx{$��  �          A��@����(��%��z�C��@���ә�������{C��                                    Bx{$�r  �          A��\@����R�#����
C�}q@����z����\����C��                                    Bx{$�  
J          A��R@Ǯ��=q�*{��ffC��@Ǯ��p����� p�C��3                                    Bx{$��  �          A�
=Aff��\)�5G�����C���Aff�Ʌ������C��)                                    Bx{$�d  "          A�
=@�������4����(�C���@�����(���
=�\)C�xR                                    Bx{$�
  T          A��A����ff���H�d(�C�L�A���أ��b�H��C�H                                    Bx{$�  "          A��@������p��^=qC�ff@����(��c\)��G�C��)                                    Bx{$�V  �          A��@�{��
=�����
C���@�{�ٮ�r�\���HC�'�                                    Bx{$��  �          A��\@�G������z�����C��q@�G����
��
��(�C�p�                                    Bx{%�  �          A�z�@�����
����Q�C�@ @���ң���33��=qC��                                    Bx{%H  D          A�=q@�p���
=�=����HC���@�p�������  ���C�                                    Bx{%+�  �          A���@�z�����%G�����C��)@�z���(�����p�C���                                    Bx{%:�  �          A���@���
=�#�����C���@��Ӆ��
=��p�C�XR                                    Bx{%I:  T          A��@ҏ\��\)�������C�@ @ҏ\�֣���(����HC���                                    Bx{%W�  �          A��@�Q����
�\)��(�C��=@�Q���(��x(���ffC�7
                                    Bx{%f�  T          A�
=@�  ���=q��G�C��=@�  ��  �v�\��G�C�5�                                    Bx{%u,  
�          A��H@��R��(��G����RC��)@��R��33�q����C�"�                                    Bx{%��  "          A�p�@�ff���������C��
@�ff����qG���C��                                    Bx{%�x  �          A��@�����H��
�o\)C�0�@�������ip���p�C��                                    Bx{%�  "          A�G�@y������33�>�HC��
@y������Q����  C���                                    Bx{%��  �          A��@�=q��G���Q��C�C���@�=q����S����
C�E                                    Bx{%�j  �          A���@��R������G��S33C�\@��R�����Z�H���HC�t{                                    Bx{%�  �          A�p�@ȣ���33�=q�s�
C�� @ȣ���G��j=q��ffC�c�                                    Bx{%۶  "          B G�@\��� (��g�C��@\��Q��e���=qC�q                                    Bx{%�\            B (�@�\)��  �����`z�C���@�\)����aG��У�C���                                    Bx{%�  
�          B 33@�\)���H���eG�C�@�\)���
�c33��z�C���                                    Bx{&�  T          B {A�R�ޏ\�X����ffC�/\A�R��z�������C�J=                                    Bx{&N  T          B p�@�\)��Q��4(����C�G�@�\)��  �����{C�R                                    Bx{&$�  
�          B ��@��
���������C���@��
��(��tQ���33C��                                    Bx{&3�  �          B ��A z���33�B{��\)C��{A z��˙�����\)C��H                                    Bx{&B@  �          B ��@�����*ff���RC���@����z����H����C��f                                    Bx{&P�  �          B �\@�=q����z���z�C�&f@�=q�ܣ��rff����C��{                                    Bx{&_�  T          B �@��
��33��H�k�C�#�@��
��{�e���=qC��f                                    Bx{&n2  �          B �@�(�������R�V�\C�q@�(���R�[
=�ɅC��{                                    Bx{&|�  �          B �R@����Q��ۅ�D��C��=@����p��R{���RC�33                                    Bx{&�~  �          B �@�����33��=q��C�>�@�����z��?
=��Q�C���                                    Bx{&�$  
�          B ��@θR��\)�����(�C��)@θR���6�R��  C�1�                                    Bx{&��  �          B �H@�ff���
�ƸR�1C��H@�ff��(��H����p�C���                                    Bx{&�p  �          B �R@�(�������H�5C�8R@�(����J�R��p�C��                                     Bx{&�  d          B @�Q���
=��  �$��C��)@�Q���(��A����HC��                                    Bx{&Լ  
�          B �@�p���G���33��C�H@�p���
=�;\)����C�=q                                    Bx{&�b  �          B ��@�ff������\��C�g�@�ff��\�=����C���                                    Bx{&�  
�          B �@�33��{�ʏ\�5G�C���@�33��ff�I�����C��\                                    Bx{' �  "          B
=@�
=�����=q�C\)C�t{@�
=��33�Pz����\C���                                    Bx{'T  �          B
=@����p���z���HC���@������7�����C��                                    Bx{'�  
�          B�@��������=q�
=C���@�����R�=�����C���                                    Bx{',�  T          B�@��\��G���  �9C���@��\��p��K����C�.                                    Bx{';F  �          B{@�
=��
=��{�N=qC�y�@�
=��(��UG���G�C��q                                    Bx{'I�            B
=@��
�����  �AG�C�\)@��
���N�\���RC���                                    Bx{'X�  
�          B �H@����G��׮�AG�C���@����33�M���ffC��                                    Bx{'g8  
�          B �@���������  � ��C���@�����
=�,z�����C��                                    Bx{'u�  �          B @�33��Q���33�6=qC�q@�33�����H  ���HC�q�                                    Bx{'��  �          B �@�  ��  �����X(�C��@�  �����YG���\)C��                                    Bx{'�*  
(          B �H@�z���{��ff�8��C�` @�z���\�I����C���                                    Bx{'��  
�          B �H@ڏ\��
=�\�.=qC�<)@ڏ\��(��B�R����C���                                    Bx{'�v  
�          B ��@Å��{��Q��%�C��3@Å���=��\)C��                                    Bx{'�  
�          B G�@�  ��  ���
�ffC�Q�@�  ��Q��8z����\C���                                    Bx{'��  T          B 33@�����  ��Q���
C���@��������2�\��
=C��\                                    Bx{'�h  T          A��@�\)�����h���љ�C���@�\)����G���\)C���                                    Bx{'�  T          A��@�33��
=�dz���p�C�e@�33��Q��(���=qC���                                    Bx{'��  
�          B (�@�G���Q��Q���(�C�R@�G���  ��
���C�E                                    Bx{(Z  
�          B �@z=q��p��%��z�C��@z=q��Q��G���{C��f                                    Bx{(   v          A��@l(���G������C�S3@l(���\�	���y��C�q�                                    Bx{(%�  
j          A��@p  ����   ��\)C�aH@p  ��(��\)�}�C��H                                    Bx{(4L  �          A���@�ff��z��%����
C��=@�ff��p��(��~�RC��                                    Bx{(B�  
�          A�
=@~�R�����H�����C���@~�R���z�����C��                                     Bx{(Q�  T          A�\)@�  ���H������G�C�c�@�  ��� �����HC��                                    Bx{(`>  �          A�33@�G���p���  ����C��
@�G�����$  ���C�@                                     Bx{(n�  
Z          A���@��R��33�r�\���HC���@��R��ff�G����
C�1�                                    Bx{(}�  �          A���@�������z�H��\C��@������
=����C�Y�                                    Bx{(�0  T          A���@��H���\������C��H@��H����#�
��=qC�)                                    Bx{(��  �          A��H@ҏ\��ff���R�(�C��@ҏ\��Q��*=q��(�C�T{                                    Bx{(�|  
�          A��R@��
��{��  �{C�J=@��
��Q��&�\���RC��R                                    Bx{(�"  �          A�33A z���  �*=q��G�C�P�A z�����	���zffC��3                                    Bx{(��  �          A�Q�@�(�����������RC�ff@�(������(����p�C��=                                    Bx{(�n  
Z          A�ff@�  ����Fff��33C��@�  ��(������ffC�(�                                    Bx{(�  �          A�=q@�33��G��0����\)C�>�@�33��Q��33�~=qC�xR                                    Bx{(�  
�          A��@�����{�j=q���
C�7
@����������{C�z�                                    Bx{)`  
�          A�\)@׮���j=q��(�C�0�@׮��p��Q���{C�s3                                    Bx{)  v          A��@���33�c�
��ffC��@������\��(�C��                                    Bx{)�  2          A���@�\)��p��xQ�����C�h�@�\)���H�����HC���                                    Bx{)-R  T          A��@��R����l(���p�C�s3@��R��p�����ffC��                                    Bx{);�  �          A���@�ff�����]p���Q�C�� @�ff��R����p�C���                                    Bx{)J�  �          A���@�=q����8Q����RC���@�=q��{�����
=C��=                                    Bx{)YD  T          A�{@����G������=qC�9�@����ff��p��333C�N                                    Bx{)g�  T          A�(�@\)����@�\@l(�C��@\)��G���G����C�
=                                    Bx{)v�  �          A��@<(����@   @�Q�C��=@<(�����dz���ffC���                                    Bx{)�6  �          A��@J=q��p�?�{@;�C�޸@J=q��\)���� Q�C��                                    Bx{)��  �          A��@����ff�8Q쾣�
C��H@�������z��2ffC��R                                    Bx{)��  �          A�\)@������
���AG�C���@������������\Q�C��                                    Bx{)�(  �          A��H@�33���\����\)C��@�33��\�=q�n�HC�>�                                    Bx{)��  T          A�z�@[�����?J=q?�Q�C�  @[���p���p���HC�*=                                    Bx{)�t  
�          A�G�@Mp����H@�
@�{C��@Mp����
�j�H��p�C��\                                    Bx{)�  
�          A��\?�G����?�p�@eC�` ?�G���  ��Q��陚C�b�                                    Bx{)��  
�          A�Q�����H@���A%p�C��{����þ�z��C��q                                    Bx{)�f  �          A�zῑ���  @�  A(  C��쿑���=q�W
=��p�C��
                                    Bx{*	  
�          A���s33����@���Az�C�  �s33��G����%�C�"�                                    Bx{*�  T          A�\)��Q���p�@j=q@�C�����Q����\�����C��=                                    Bx{*&X  �          A�G���=q��\)@b�\@�\)C��Ϳ�=q��=q�Q����HC��                                    Bx{*4�  T          A��ÿc�
���@@��@�Q�C�.�c�
�����9�����C�.                                    Bx{*C�  T          A��H�^�R���
@(Q�@��C�4{�^�R��\)�QG����C�33                                    Bx{*RJ  "          A��\�&ff��p�@6ff@�\)C�ff�&ff��G��A����C�ff                                    Bx{*`�  T          A�z�\��\)@1�@��HC��f�\����Fff��C��f                                    Bx{*o�  T          A�=q��\)��\)@)��@�33C��\��\)���H�N{���C��\                                    Bx{*~<  T          A���?
=��G�@Mp�@�(�C��=?
=����*=q���
C��=                                    Bx{*��  �          A��R?fff����@e�@љ�C��3?fff�������C���                                    Bx{*��  �          A��H?Q���@���A(�C�?Q����\���ÿ��HC��                                     Bx{*�.  �          A�ff?������@��
A�\C��?����{�5���C�                                      Bx{*��  �          A�ff?�ff���@��HA�HC���?�ff��  ���R���C��
                                    Bx{*�z  �          A�ff?��
���@�ffA�RC��{?��
��  ��\)� ��C��3                                    Bx{*�   �          A�Q�?������@�(�A�HC���?����{�.{���RC��R                                    Bx{*��  �          A�@,����ff@j�H@�Q�C���@,�������ff�w
=C�~�                                    Bx{*�l  
�          A�=qA���񙚿�{��C��HA�����
��p��E�C��                                    Bx{+  T          A���@������H�W
=�\C�/\@�����ff����*�HC�P�                                    Bx{+�  �          A��@����G�?W
=?��C��=@����Q���G����C��q                                    Bx{+^  T          A�p�@�(���33>��?B�\C�` @�(�����ff�p�C�w
                                    Bx{+.  "          A��@�����?��
@ffC�=q@���G���z��G�C�J=                                    Bx{+<�  
Z          A�  @���{@(�@��C��@���p��P  ��\)C���                                    Bx{+KP  
�          A�\)@��
���@<��@�C��@��
���
�/\)����C��                                    Bx{+Y�  "          A�p�@�33��(�@!G�@�(�C�@�33����J�H���HC��                                    Bx{+h�  �          A�=q@�\)��=q<�=L��C�E@�\)��(������$(�C�^�                                    Bx{+wB  
          A��@�G���\)<�=L��C���@�G���\)��  �#�C���                                    Bx{+��  �          A���@��
����?��H@K�C��@��
��\)�xQ���\)C���                                    Bx{+��  "          A��
?�����G�@}p�@�33C�Y�?������H��(��K�C�W
                                    Bx{+�4  �          A�?ٙ���ff@�A
=C��R?ٙ�����z�H����C��{                                    Bx{+��  T          A��@"�\��Q�@��
A�C�aH@"�\���׿��\�
=C�\)                                    Bx{+��  
�          A��@z���@���A
=C�,�@z���=q������\C�&f                                    Bx{+�&  T          A��\@���G�@���A33C��q@�������G�C��R                                    Bx{+��  �          A�ff@"�\��33@���@�\)C�b�@"�\��\)��=q�{C�]q                                    Bx{+�r  
�          A�ff@333���@�33@���C���@333��
=��  �1�C��q                                    Bx{+�  T          A�(�@n{���
@�p�A  C���@n{��=q��33���C�y�                                    Bx{,	�  �          A�ff@e���(�@�AQ�C�^�@e���\�����C�U�                                    Bx{,d  
�          A��\@g���  @�{A(�C�h�@g����R�^�R��\)C�^�                                    Bx{,'
  �          A�=q@aG���  @�AQ�C�P�@aG���ff�����ffC�G�                                    Bx{,5�  
�          A�{@j�H��ff@vff@�{C�t{@j�H��녿ٙ��J�HC�o\                                    Bx{,DV  
�          A�{@�{��=q@\��@�ffC���@�{��33���xQ�C���                                    Bx{,R�  �          A��@�p�����@y��@陚C�+�@�p���׿�\)�AG�C�#�                                    Bx{,a�  
Z          A��@e���R@���A33C�e@e��󙚿@  ���C�Y�                                    Bx{,pH  
�          A��@R�\���@�z�A�
C�)@R�\�������
=C�{                                    Bx{,~�  �          A�
=@J=q��33@�33A�RC��q@J=q����z��
�HC���                                    Bx{,��  �          A���@Z�H���H@��HAffC�=q@Z�H��33����C�5�                                    Bx{,�:  "          A�\@QG���R@�Q�A (�C�R@QG����ÿ�p���
C��                                    Bx{,��  �          A��R@S33����@�ff@�(�C��@S33�����ff��C�R                                    Bx{,��  �          A��R@HQ����\@���A  C���@HQ���\)�W
=����C��                                    Bx{,�,  
�          A�ff@,���@�(�A"{C���@,����p����;B�\C���                                    Bx{,��  T          A�ff?��H���@�z�A"�RC��q?��H����Q�.{C���                                    Bx{,�x  
�          A�ff@�\��(�@�G�A�C���@�\���
�B�\��33C��=                                    Bx{,�  �          A�  ?�ff��{@��
A�RC��R?�ff����33�(��C���                                    Bx{-�  T          A�@  ��p�@�ffAG�C�'�@  ��
=������HC��                                    Bx{-j  
�          A�@����@�  A
=C�9�@����;L�;�p�C�0�                                    Bx{-   �          A��@!�����@�=qAC�k�@!���=q��p��0��C�b�                                    Bx{-.�  
�          A��H@(������@�33A33C��f@(�����
�zΉ�C�}q                                    Bx{-=\  
�          A�R@1G����@�
=A\)C��f@1G��񙚿333����C���                                    Bx{-L  
Z          A�Q�@z�����@�ffA
=C�8R@z���p��xQ��C�1�                                    Bx{-Z�  
�          A�(�@33��R@���A	C�5�@33��\)�c�
��
=C�/\                                    Bx{-iN  �          A�@"�\��33@�ffA�RC�q�@"�\���H�B�\��Q�C�h�                                    Bx{-w�  
�          A�\)?������@���A)��C�Ф?�����H>#�
>��RC�Ǯ                                    Bx{-��  �          A�G�@���G�@�G�A�C��@���׾�33�(��C��                                    Bx{-�@  �          A�G�@W����H@��A33C�@ @W����O\)���
C�7
                                    Bx{-��  
          A�p�@�����(�@�G�@�{C���@�����{��ff��RC���                                    Bx{-��  
�          A��H@�����@���A�\C�'�@����=q����G�C�q                                    Bx{-�2  "          A���@����z�@�z�AG�C�Z�@����{�8Q쾮{C�J=                                    Bx{-��  
(          A�ff@�������@��\A+33C�Ǯ@�����33>�z�?��C���                                    Bx{-�~  "          A�Q�@��\���@���A8��C��R@��\����?@  ?�
=C�޸                                    Bx{-�$  
(          A�Q�@����癚@�(�AL  C�p�@�����\)?�{@%C�U�                                    Bx{-��  "          A�  @����
=@�=qAJ�RC���@�����?���@!G�C�z�                                    Bx{.
p  
Z          A�(�@�G���\@��
A<Q�C�33@�G��홚?W
=?���C�)                                    Bx{.  "          A�  @�����@��AJ{C�S3@�����33?�ff@�RC�9�                                    Bx{.'�  �          A�{@����@ᙚAY��C�]q@���
=?�ff@\(�C�@                                     Bx{.6b  �          A��
@�ff��{@�A]C�ff@�ff���?�Q�@mp�C�G�                                    Bx{.E  
�          A�33@~{��@�=qAC33C��@~{����?��@�C���                                    Bx{.S�  T          A�
=@j=q��\@��HA4z�C��
@j=q��33?�?��C��f                                    Bx{.bT  �          A��H@mp���@ʏ\AC�
C��f@mp�����?���@�
C���                                    Bx{.p�  T          A���@p����  @���AZ{C���@p����ff?�ff@\��C��                                     Bx{.�  E          A��@�33���@��An{C�XR@�33��33@p�@��C�5�                                    Bx{.�F  
�          A�ff@��\���
@�(�Amp�C�S3@��\��
=@(�@�ffC�0�                                    Bx{.��  "          A�Q�@�G���@��RAp  C�J=@�G���
=@!G�@�33C�'�                                    Bx{.��  �          A��@��\��Q�@�Aap�C��@��\����@�\@{�C��3                                    Bx{.�8  
�          A��@j=q��@�
=Ap��C��=@j=q��
=@!�@�(�C���                                    Bx{.��  
�          A�\)@`  ���@�(�AVffC�|)@`  ��G�?�@N{C�e                                    Bx{.ׄ  
(          A�
=@n�R��R@��
AVffC��R@n�R����?�@N�RC��                                     Bx{.�*  "          A��@�����
A�
A���C��@����\@�{Ap�C���                                    Bx{.��  �          A��@w�����@��Ak�C��@w����@ff@���C��                                    Bx{/v  �          A�(�@x�����
@�Aqp�C���@x����33@!�@�C���                                    Bx{/  �          A��
@�  ��G�@�{Ar{C�
=@�  ���@#33@�
=C��=                                    Bx{/ �  
�          A�\)@�\)����A2{A��C���@�\)��G�@�p�AAC�y�                                    Bx{//h  �          A�R@�������A$��A��HC�9�@�����(�@��A'33C�                                    Bx{/>  T          A�=q@r�\��ffA�RA�{C��=@r�\���@��
Ap�C��H                                    Bx{/L�  "          A��@j=q��Q�A�A�=qC�� @j=q��\@N�R@�33C��q                                    Bx{/[Z  �          A陚@dz���
=A�
A�\)C��@dz���  @k�@�\)C���                                    Bx{/j   �          A���@'
=�޸RAp�A��C���@'
=��R@@  @�p�C���                                    Bx{/x�  �          A�ff@���  @��AiG�C�y�@�����@��@��HC�ff                                    Bx{/�L  
�          A�{@{��
=@�{A,��C�}q@{���>�z�?z�C�q�                                    Bx{/��  �          A��@G�����@��A0��C��@G���G�>��?O\)C���                                    Bx{/��  
�          A�?�����
@��A
=C��?�����þ�녿Q�C�޸                                    Bx{/�>  T          A��?�p���\@n{@���C��)?�p���=q��Q��6ffC��R                                    Bx{/��  �          A�R?޸R�ҸRA8��A�G�C���?޸R���
@�AVffC�Ǯ                                    Bx{/Њ  T          A�G�?������
A@z�A��HC��H?����߮@���Aep�C�j=                                    Bx{/�0  
�          A�
=>k���33AC
=A��C�AH>k���G�@�\Ak\)C�=q                                    Bx{/��  "          A�z�?Q����A(�A�=qC��q?Q��㙚@�  A�
C��3                                    Bx{/�|  
�          A��
?E���p�@�\Au�C�˅?E���R@!G�@���C��                                    Bx{0"  T          A�\)?����\)@陚Al(�C�aH?����=q@�R@�
=C�W
                                    Bx{0�  �          A噚?�z���
=@��
Av�\C�t{?�z���ff@#�
@��
C�h�                                    Bx{0(n  T          A�p�?�{��z�@�33A]�C�ff?�{��R?�G�@a�C�]q                                    Bx{07  
�          A��?����H@���As�
C�5�?���{@{@�{C�,�                                    Bx{0E�  
i          A�Q�?n{�ڣ�A
=A�p�C���?n{����@J�H@��
C��                                    Bx{0T`  
�          A��?Y�����
Ap�A�{C��?Y����=q@Tz�@�ffC��)                                    Bx{0c  
�          A�=q?:�H��{AA�Q�C��?:�H��\@U@�
=C��q                                    Bx{0q�  
�          A��?Q�����A�
A��HC�޸?Q����
@o\)@�G�C���                                    Bx{0�R  
�          A�(�?�\)�ٮA=qA�
=C�q�?�\)��(�@XQ�@ٙ�C�c�                                    Bx{0��  4          A�  ?����ff@��HA��C���?����\)��\���C���                                    Bx{0��  �          A�  >����@�G�Af�\C��H>���@   @��C�}q                                    Bx{0�D  
�          A�녾���ڸR@��RA��C��H�����\@:�H@�z�C��                                    Bx{0��  �          A�녾�
=�ڸR@�{A{\)C��\��
=��(�@)��@�33C��3                                    Bx{0ɐ  �          A�Q�>#�
�߮@��
A5�C�*=>#�
��(�?   ?�G�C�*=                                    Bx{0�6  �          A�녿G���p�@�  At��C�/\�G����@(�@�C�5�                                    Bx{0��  �          A�  ��  ��G�A-��A�(�C��ᾀ  �݅@��RABffC���                                    Bx{0��  �          A�z�>�ff��{Ao\)A�G�C���>�ff��ffA%��A�=qC�|)                                    Bx{1(  �          A�
=>�(���  A�A��C�z�>�(���{@�
=A�C�t{                                    Bx{1�  �          A��>�{�޸R?��H@@��C�Y�>�{��33�e��z�C�Y�                                    Bx{1!t  �          A���>����z�@���AzffC�E>����  @%@��C�B�                                    Bx{10  �          A�(�?�\��=q@�ffAr=qC���?�\��\)@ff@�\)C���                                    Bx{1>�  �          A�=q>�����@�\Av�\C�|)>����G�@�R@��C�xR                                    Bx{1Mf  �          A�{>�(�����@�Ag33C�q�>�(���p�@   @���C�n                                    Bx{1\  �          A�  ?Y���܏\@��Ah��C��H?Y����G�@33@��
C�ٚ                                    Bx{1j�  �          A�=q?333��=q@�{Aq�C���?333��\)@�@�C��{                                    Bx{1yX  �          A�=q>���ۅ@��A~=qC�~�>����33@-p�@�{C�z�                                    Bx{1��  �          A�{�u��(�A��A��C��u���@N{@�\)C���                                    Bx{1��  �          A�{�����ٙ�A��A�C��������ff@^�R@�  C��                                    Bx{1�J  �          A��^�R��=qAG�A�p�C��^�R���\@���A\)C�q                                    Bx{1��  �          A��Y�����HAG�A��C�R�Y�����H@���A
�\C�#�                                    Bx{1  �          A��@  ����Ap�A�G�C�4{�@  ����@�G�A
�RC�=q                                    Bx{1�<  �          A�(��aG���=qA��A�  C��׾aG����
@~{A   C��f                                    Bx{1��  �          A�=q�L������Az�A�(�C�%�L����G�@�
=A(�C�/\                                    Bx{1�  �          A�  ��{�י�Az�A�  C��׾�{��@�ffA�C���                                    Bx{1�.  �          A�  =#�
�י�AQ�A�C��=#�
��@�A
=C�
=                                    Bx{2�  �          A��
�0����A�A�\)C�C׿0�����\@�Q�A{C�L�                                    Bx{2z  �          A�녿   ��G�A z�A���C�xR�   ��ff@�\)A!�C�~�                                    Bx{2)   �          A�  �G���
=A!��A�  C�*=�G���=q@�G�A#33C�5�                                    Bx{27�  �          A��
�����\)A)G�A�z�C�� �����G�@���A3�
C��\                                    Bx{2Fl  �          A㙚�s33�ӅA'�A��RC��R�s33��G�@�A0(�C�f                                    Bx{2U  �          A���G���33A(��A�=qC�����G����@���A3
=C��
                                    Bx{2c�  �          A�p����хA0  A�{C��
����(�@��AB�RC��=                                    Bx{2r^  �          A�\)������A((�A��C��쿋�����@��RA1G�C��                                    Bx{2�  �          A��
����p�A
=A�p�C�o\����z�@�33A��C�w
                                    Bx{2��  T          A��
���ՙ�AffA��RC�z�����\@���A33C��H                                    Bx{2�P  �          A��G����A%A��HC�'��G��ߙ�@�G�A+\)C�33                                    Bx{2��  �          A��G���=qA.=qA�  C�&f�G�����@��HA=p�C�33                                    Bx{2��  T          A��
�B�\��
=A+�A���C�,ͿB�\��G�@���A7
=C�8R                                    Bx{2�B  T          A���p���G�A2{A�(�C�b���p���(�@��HAE��C�y�                                    Bx{2��  �          A�녿�33��Q�A7\)A��C�t{��33��@�p�APz�C���                                    Bx{2�  �          A������{A?
=Aƣ�C�/\�����=q@�{Ab{C�L�                                    Bx{2�4  �          A㙚��33��\)A:�HA��C��ΐ33���@���AXQ�C��\                                    Bx{3�  �          A���ff��z�A?33A�z�C����ff�ܣ�@�AaG�C��f                                    Bx{3�  �          A��u��\)A:ffA��C���u���@ӅAW
=C��                                    Bx{3"&  �          A�p��k���33A:�HA�(�C��)�k�����@�z�AX  C��                                    Bx{30�  T          A�G��������HAC\)A˙�C�aH�����ۅ@�ffAj�HC�}q                                    Bx{3?r  �          A��H�\���AEG�A�(�C�K��\����@��HAp  C�h�                                    Bx{3N  �          A��Ϳٙ��ʣ�AI�A�p�C�3�ٙ���  @���AzffC�5�                                    Bx{3\�  �          A�z��\)��=qAJ{A�  C�*=��\)�ٮ@��A{33C�K�                                    Bx{3kd  �          A�(�� ���ȣ�AN=qA���C���� ����z�@�{A��\C��                                     Bx{3z
  T          A�(���Q��ɮAJffAԏ\C����Q����@�p�A|  C���                                    Bx{3��  
�          A�  �Ǯ�ɮAJ{Aԏ\C�<)�Ǯ���@���A{�C�\)                                    Bx{3�V  �          A���G���ffAF�\A��HC��)��G��م@��As�C��                                    Bx{3��  �          Aᙚ��33�ɮAH��AӅC��쿓33��
=@��Ax��C���                                    Bx{3��  �          Aᙚ���H��p�AJ=qA���C�� ���H����@�z�A{�C���                                    Bx{3�H  �          A��������AM�A�(�C�������أ�@��\A���C���                                    Bx{3��  �          A�p���Q����AQ�A�  C�Y���Q���p�AffA��\C�y�                                    Bx{3��  �          A�
=�˅��{AT  A���C�(��˅�֏\A��A�p�C�L�                                    Bx{3�:  �          A�R��G����HAW\)A�
=C�>���G���Az�A�p�C�aH                                    Bx{3��  �          A�R�������AZ�\A���C�*=������33A�
A�
=C�P�                                    Bx{4�  �          A�Q��p���=qA^�HA�=qC��ÿ�p����A��A�ffC�!H                                    Bx{4,  �          A�(���=q��G�A`��A���C��ÿ�=q��G�A�HA�
=C�                                    Bx{4)�  �          A���{��  Ad  A���C�Ǯ��{��=qA=qA��HC��R                                    Bx{48x  T          A߮��{��\)Ae��A��C�Ǯ��{���
A  A��HC���                                    Bx{4G  �          A�G����H���Aip�A�(�C�����H�Џ\A(�A��
C���                                    Bx{4U�  �          A�������HAd��A���C��{����G�A33A�z�C�                                    Bx{4dj  �          A��H�\)��(�Al(�A��C�Ff�\)��\)A33A�p�C��f                                    Bx{4s  �          Aޣ�������Ai��A�33C�|)����Az�A��\C���                                    Bx{4��  �          A�ff��p���
=Ah  A��C��)��p����A�\A���C��3                                    Bx{4�\  �          A�녿�=q��p�Ak�
B p�C����=q�θRA�HA��
C���                                    Bx{4�  �          Aݮ��\����Al(�B �C��H��\��=qA33A�=qC��)                                    Bx{4��  �          A�
=���H����AjffB   C������H��(�AG�A���C��                                    Bx{4�N  �          A��Ϳ������
Alz�B�C��H������G�A�A���C��
                                    Bx{4��  �          A�ff���H���HAmB��C��{���H�̏\A!G�A���C��\                                    Bx{4ٚ  �          A�Q��(����RAn{B��C�����(���ffA!p�A��
C�˅                                    Bx{4�@  �          A�{�
�H��(�An=qB�C�O\�
�H���A!A�z�C���                                    Bx{4��  
�          Aۮ�  ����An�\B�\C�4{�  ��\)A"{A�33C�y�                                    Bx{5�  �          A�\)�G���As
=B�C�'��G���  A'
=A�
=C�o\                                    Bx{52  �          A����\)���Av�RB	=qC�(��\)�ȸRA+\)A�=qC�s3                                    Bx{5"�  �          A�z��	����G�Aw\)B	�C�B��	����{A,  A�p�C���                                    Bx{51~  �          A�{�ff��  AyG�Bp�C��q�ff��
=A.=qA�ffC�N                                    Bx{5@$  �          Aٮ�0�����A|��B�
C�l��0����p�A2{A�33C��\                                    Bx{5N�  �          A�p��.{��A|(�B��C�xR�.{��G�A1A���C�ٚ                                    Bx{5]p  �          A���{���HAx��B�RC����{��  A-��A��\C�'�                                    Bx{5l  �          AظR�*=q���\Aw�BffC����*=q�ř�A,��A��
C��                                    Bx{5z�  �          A�ff�<(�����Au�B
ffC�4{�<(��ŅA*�RA�C���                                    Bx{5�b  �          A�  �g����At��B	��C�P��g���z�A)A��C��                                    Bx{5�  �          A�  �����Q�A|  B�RC�˅�����  A1G�A��C�%                                    Bx{5��  �          A�{�����\)A�(�B�\C��׿����p�A5�A��C��                                    Bx{5�T  �          A�  ��\)���
A~�RB�C�����\)��A4  A���C�Ф                                    Bx{5��  �          A�p���
=����A}G�B  C�uÿ�
=��p�A2ffA�C���                                    Bx{5Ҡ  �          A����;���\)A33B�C���;����A5�A�\)C��=                                    Bx{5�F  �          A֣��5���Q�A��\B33C�8R�5����RA733A�(�C���                                    Bx{5��  �          A�=q�)����A���B
=C�q��)����Q�A8(�AˮC���                                    Bx{5��  �          A���3�
����A���B(�C�33�3�
��\)A9A��
C��f                                    Bx{68  �          A�G��:�H��(�A��B
=C���:�H����A8��A�p�C���                                    Bx{6�  �          A�G��2�\����A�  BG�C�4{�2�\��z�A:�RAϙ�C���                                    Bx{6*�  �          A���'�����A��B�C�h��'����
A=�AҸRC��R                                    Bx{69*  �          A����*�H���A�B�C�S3�*�H��G�A>�\Aԏ\C��f                                    Bx{6G�  �          Aԏ\�6ff���
A���B�C�R�6ff���A<��A��HC���                                    Bx{6Vv  �          Aԏ\�.�R��p�A���B  C�=q�.�R���HA=�A�ffC��3                                    Bx{6e  T          A�=q�S�
����A�33B�C�o\�S�
��(�A=G�A��
C���                                    Bx{6s�  T          A�  ������A�B�
C��
����Q�A>=qAՅC�                                      Bx{6�h  �          AӅ�$z���G�A�Q�B33C�t{�$z����\A;33A�(�C���                                    Bx{6�  �          A�\)�ff��=qA��B(�C��)�ff���
A>{A�C�%                                    Bx{6��  �          A����  ����A�
=Bz�C��R�  ����AAG�A�(�C�>�                                    Bx{6�Z  �          Aҏ\���Q�A���Bp�C��
���=qA@��A��C�"�                                    Bx{6�   �          A�(���\��\)A�\)B��C�����\��p�AB{A�  C�/\                                    Bx{6˦  �          A��
�p����A�33B ffC����p���{AF=qA�p�C�AH                                    Bx{6�L  T          AѮ��G���p�A�G�B �
C�z��G���(�AFffA��C�Ф                                    Bx{6��  �          A�G��G�����A�\)B!(�C�
�G����AF�\A�z�C�z�                                    Bx{6��  �          A�
=�
=q��z�A��B$�\C��R�
=q���AK�A�33C�G�                                    Bx{7>  �          AиR��
=���A��HB)�
C�  ��
=��33AS\)A�p�C��=                                    Bx{7�  �          A�(��\)��Q�A�=qB&p�C�Q��\)���
AM��A�RC��
                                    Bx{7#�  �          A�p��0  ����A�{B$
=C��{�0  ��{AH��A��
C���                                    Bx{720  
�          A�33�1���(�A���B%
=C����1���p�AJ=qA�C�w
                                    Bx{7@�  �          A��H�4z�����A��HB(�RC����4z����AO�A���C�`                                     Bx{7O|  �          AΏ\�"�\����A���B,{C�#��"�\��{AT(�A�G�C��
                                    Bx{7^"  �          A�z��6ff��=qA��
B-�\C��H�6ff����AVffA�=qC�J=                                    Bx{7l�  �          A�=q�*=q���RA�G�B,��C���*=q��\)AT��A���C��=                                    Bx{7{n  �          A��
��R��{A��B'�C�H���R���
ALQ�A�C���                                    Bx{7�  �          A�Q��!G����A�33B$�HC�g��!G���Q�AO�A��C���                                    Bx{7��  �          A�33�ff��A��
B$
=C����ff���AL��A�  C�)                                    Bx{7�`  �          A�
=��
���\A���B"��C��R��
���\AJffA��HC�,�                                    Bx{7�  �          Aՙ�� ������A��B#\)C�&f� ������AK�
A�  C���                                    Bx{7Ĭ  �          A��H��Q���p�A�(�B$C�7
��Q����
AMG�A�\C���                                    Bx{7�R  �          A�
=���
���
A�{B!�\C��ÿ��
��AG�
A��
C�"�                                    Bx{7��  �          A��ÿ��
���
A��RB �RC��)���
��  AH(�A��
C�'�                                    Bx{7�  �          A�
=������A��HB%�\C�J=������
AN{A�p�C��                                    Bx{7�D  �          A���=q��{A�{B%�C�\)��=q���RAL��A�C��)                                    Bx{8�  �          A�
=���R��  A���B!z�C�'����R���
AEG�A���C���                                    Bx{8�  �          A�33���\��A�B"�
C�/\���\���AG33A��C�p�                                    Bx{8+6  �          A���>�Q���33Ad(�B{C�w
>�Q����
AG�A�G�C�j=                                    Bx{89�  �          Aң�>�����p�AU�A�{C�k�>�����ffAG�A�
=C�aH                                    Bx{8H�  �          A�ff?0�����AG
=A��C�ٚ?0����z�@��AyG�C�Ǯ                                    Bx{8W(  �          A�=q?=p���(�A;33A�p�C��?=p���(�@ǮA[�
C���                                    Bx{8e�  �          Aҏ\?�����
A5G�A�  C�Ff?�����@��\AL��C�.                                    Bx{8tt  �          A�33?����  A7�A�Q�C�G�?���ͅ@��RAP��C�/\                                    Bx{8�  �          AӮ?�����A=�A�=qC�b�?����\)@ə�A\Q�C�Ff                                    Bx{8��  �          A�ff?�����HAA�A�z�C�B�?���͙�@�G�Ad(�C�(�                                    Bx{8�f  �          A��
?O\)���AN{A��C�  ?O\)��\)@��A�  C���                                    Bx{8�  �          AӮ?��
���AT  A�=qC�H�?��
��{@�=qA�p�C�*=                                    Bx{8��  �          A���?������\AK�A�RC���?����ʣ�@�  A\)C�~�                                    Bx{8�X  �          A�
=?������AM�A�(�C���?����{@��A���C��f                                    Bx{8��  �          A�  ?����33A/�
A�z�C�
?����Q�@�AR=qC��                                    Bx{8�  �          A�  ?�G����HA1G�A�ffC��?�G���=q@���AUp�C��f                                    Bx{8�J  �          A�?��
��\)A-�A�z�C��{?��
��ff@�G�AM�C��                                    Bx{9�  �          A�\)?�{��A*=qA�(�C��?�{��Q�@�G�AD(�C��                                    Bx{9�  �          A���?˅��Q�A&{A�\)C��?˅��ff@�Q�A9�C��q                                    Bx{9$<  �          A���?��H���HA,��AͮC��q?��H���
@�{AJ{C��R                                    Bx{92�  �          A��?fff���
A:ffA��C�,�?fff��z�@��HAl(�C�3                                    Bx{9A�  �          A�p�?Tz���(�AB{A���C��?Tz����@�33A33C�                                      Bx{9P.  �          A��?^�R����AD��A��HC�&f?^�R���H@�G�A�\)C��                                    Bx{9^�  �          A�
=?z�H��=qAF�HA�C�L�?z�H��z�@�p�A��C�,�                                    Bx{9mz  �          A���?����{AG33A�Q�C�c�?����Q�@�{A�ffC�AH                                    Bx{9|   �          AƸR?������AC�A��C�aH?������@�{A��C�@                                     Bx{9��  �          A�z�?����z�AC\)A�  C�g�?����Q�@�p�A��C�Ff                                    Bx{9�l  �          A�(�?��\���
A=p�A�\C�XR?��\����@�  As�C�9�                                    Bx{9�  �          A�{?u����A:{A�(�C�B�?u��p�@�Q�AjffC�&f                                    Bx{9��  �          AŮ?�  ���A4Q�A���C���?�  ��@��A[�C�~�                                    Bx{9�^  �          A�p�?�33��Q�A8Q�Aޣ�C���?�33����@�z�AfffC�b�                                    Bx{9�  �          A��?�Q���
=A3�AظRC��H?�Q����@��AZ=qC��R                                    Bx{9�  �          A�
=?�=q��z�A-p�A���C��?�=q���
@�(�AI�C��                                    Bx{9�P  �          A�
=?�ff��(�A.ffA�(�C�XR?�ff���@�{AL(�C�'�                                    Bx{9��  
�          A�33@��z�A,��A�C��3@���
@��AG
=C�|)                                    Bx{:�  �          A�G�@G�����A+\)A��C��\@G���  @��RAC
=C��3                                    Bx{:B  �          A�\)@5����A(z�A�(�C���@5���
@�Q�A;�C�aH                                    Bx{:+�  �          A�33@{���
A.�RA�ffC��@{���@��AK
=C��f                                    Bx{::�  �          A��H?Ǯ���A2=qA�\)C�
=?Ǯ��G�@�z�AS�
C�޸                                    Bx{:I4  �          A��?�  ���RA0(�Aՙ�C��R?�  ����@�  AO�C��                                    Bx{:W�  �          A�\)@z���\)A((�A��C�@z���Q�@��A<z�C���                                    Bx{:f�  �          A�\)@�R���A((�A��C�=q@�R��(�@�\)A<Q�C��)                                    Bx{:u&  �          A�
=@
�H��  A,��A�=qC�ٚ@
�H����@���AH  C��q                                    Bx{:��  �          A���@������A-G�A�33C���@����G�@��AIp�C���                                    Bx{:�r  �          A�z�@�
���\A/�A���C�@�
����@�\)APQ�C�˅                                    Bx{:�  �          A���@	�����RA333A܏\C��H@	����G�@�\)A[
=C��H                                    Bx{:��  �          A�33?�(����A<��A�(�C��?�(����@�z�Aup�C�l�                                    Bx{:�d  T          A�
=@�R���HA>�\A�RC��@�R����@�Q�AzffC��                                     Bx{:�
  �          A��@(����RA7�
A�p�C�L�@(����@���Ag�C���                                    Bx{:۰  �          A�\)@4z���p�A;33A��
C��
@4z����@�  ApQ�C�y�                                    Bx{:�V  �          A�=q@HQ���(�AUB��C�p�@HQ���\)A�A�p�C���                                    Bx{:��  �          A�{@{��AH��A��C�z�@{��\)@�\)A���C��                                    Bx{;�  �          A��@'
=���RAV�\B

=C�ٚ@'
=��=qAQ�A�p�C�aH                                    Bx{;H  �          A�(�@^{��\)Ad��B��C�C�@^{���HA(�A��C���                                    Bx{;$�  �          A���@`����33At��B 
=C��@`������A&�HA��
C���                                    Bx{;3�  �          A���@{���RAS
=B\)C��@{��  A (�A�33C�5�                                    Bx{;B:  �          A�33>�����A!�A��C�.>����@��HA4��C�*=                                    Bx{;P�  �          A���?+����A{A��RC��?+�����@u�A�
C��
                                    Bx{;_�  �          A�p�?�{���\A/33A��C�B�?�{��33@���A^=qC��                                    Bx{;n,  �          A���?�33���HA;33A��\C���?�33��33@�=qA{33C�k�                                    Bx{;|�  T          A�G�?�z����HA ��A��C���?�z����@�ffA/\)C��H                                    Bx{;�x  �          A�Q�?5��
=A�A��
C��
?5��@_\)A�
C��                                    Bx{;�  �          A�
=?�����\)A
=Aȏ\C��=?������@�{A(��C���                                    Bx{;��  �          A��
@����A+�A�ffC�o\@����
@���AUG�C��                                    Bx{;�j  �          A�G�@I�����HA733A�G�C���@I�����H@�=qAs\)C�\                                    Bx{;�  �          A�(�@L(�����AEB ffC��f@L(�����@�G�A���C�!H                                    Bx{;Զ  �          A���@c�
��\)AR�RB
�HC�c�@c�
��33A (�A�{C��
                                    Bx{;�\  �          A�@R�\��\)AI��B
=C���@R�\��  @�(�A��C�XR                                    Bx{;�  �          A�\)@QG���{ATz�B
=qC��H@QG���{A (�A��C�E                                    Bx{< �  �          A�p�@P����G�AW
=B
=C��@P����A�RA�G�C�Ff                                    Bx{<N  �          A���@���33AT��B
ffC�@ @���G�@�ffA��HC��R                                    Bx{<�  �          A��@O\)��AY��B�
C���@O\)���RA�\A�Q�C�P�                                    Bx{<,�  T          A�  ?Tz���ffA�p�B��fC�(�?Tz��3\)A�Brz�C�                                      Bx{<;@  �          A���8Q��*�HA�B�=qC�q�8Q����A�=qB�aHC�c�                                    Bx{<I�  �          A��׿Q�@|��A�
=B��=Bˊ=�Q녿��HA��B�=qCwB�                                    Bx{<X�  �          A���>�z�@�=qA�p�B��)B�B�>�z�B�\A�\)B�G�C���                                    Bx{<g2  �          A��׿c�
A"�RA�B��B���c�
@��
A��B��B���                                    Bx{<u�  �          A�ff�n{AG�A�z�B�z�B�Q�n{@^{A�  B�B�B��f                                    Bx{<�~  �          A��ÿ.{A(�A��HB�u�B�k��.{@G�A�  B�\)B�Q�                                    Bx{<�$  �          A���
=qA�HA�ffB�B�Ǯ�
=q@N�RA���B��B�#�                                    Bx{<��  �          A���\)AM�A��RBc{B��ÿ\)@�A�(�B��fB�Ǯ                                    Bx{<�p  �          A�녿�  AXz�A�33BZ��B��{��  A Q�A�=qB��=B�                                    Bx{<�  �          A�Q쿣�
AT��A�
=B]B����
@��RA���B�33B��)                                    Bx{<ͼ  �          A�ff��p�AS
=A��B_{B��3��p�@�\A�  B��Bƀ                                     Bx{<�b  �          A����33AQA��B`z�B�.��33@�ffA�G�B���B�B�                                    Bx{<�  �          A����{A-�A��By��B�G���{@�(�A�z�B�=qB�\                                    Bx{<��  �          A�zῊ=qA\)A��\B��
B��Ὴ=q@Tz�A�ffB�(�B���                                    Bx{=T  �          A��\��A�RA���B�aHB�z῕@s33A��
B�ǮB�L�                                    Bx{=�  �          A�zῬ��A(�A��
B���B�����@3�
A��RB��B�Q�                                    Bx{=%�  �          A��
�   A%p�A���B}ffB�(��   @���A��B��HB��                                     Bx{=4F  �          A�p��#33A-�A���BwffB�aH�#33@��
A�(�B�(�B�B�                                    Bx{=B�  �          A���=u@���A�ffB�.B�L�=u�	��A��RB��\C��
                                    Bx{=Q�  �          A��R�B�\@���A��B��
B�Ǯ�B�\���A�  B�ǮC�"�                                    Bx{=`8  �          A��R��{@�  A���B��qB�(���{��A�ffB��\C:=q                                    Bx{=n�  �          A��Ϳ��H@ۅA�=qB�ffB����H?   A�ffB�� C!:�                                    Bx{=}�  �          A��R���@��A��B�aHB�z����?333A�B�� C"��                                    Bx{=�*  �          A��\�G�@��A�B�
=Bգ��G�?�
=A�p�B��C�=                                    Bx{=��  �          A�Q��=q@�
=A��HB���BָR�=q?��A�
=B�k�C�                                    Bx{=�v  T          A�{��@�  A���B��B�  ��?�33A���B�C                                      Bx{=�  T          A��
� ��@���A�z�B��HB��
� ��?���A��HB��C5�                                    Bx{=��  �          A���У�@�p�A�\)B�(�B��ͿУ�?��
A��HB��=C��                                    Bx{=�h  �          A�G�����@���A�\)B��=B�(�����?�  A���B��Cٚ                                    Bx{=�  �          A�
=��G�@��A�\)B�Bǀ ��G�?c�
A���B�p�C�
                                    Bx{=�  �          A��ÿ���@�(�A���B��=B�����?uA�ffB��HC�R                                    Bx{>Z  �          A��׿�{@�33A�  B��B�녿�{?�Q�A��B�(�C�\                                    Bx{>   �          A��\��G�@�{A��
B���BȞ���G�>��A�=qB�aHCz�                                    Bx{>�  �          A�z�#�
@�G�A��HB�B�B�B��#�
=#�
A�ffB���C0�)                                    Bx{>-L  �          A�=q��@���A��B�  B��=����\A�=qB�ffCQ�                                    Bx{>;�  �          A�  ��{@�z�A���B�\B����{�\)A��B�.CJk�                                    Bx{>J�  
�          A����u@�  A�{B�=qB�\�u?E�A�G�B��C@                                     Bx{>Y>  �          A����L��@�Q�A��\B��B����L�;�z�A�p�B�B�CkY�                                    Bx{>g�  �          A����G�@���A�B�(�B�33��G��O\)A��B���C�)                                    Bx{>v�  �          A�G����@�Q�A���B��
B���׿�A�33B�Q�Cf\)                                    Bx{>�0  �          A�33��z�@��
A�(�B�(�B�#׾�zῠ  A��HB���C���                                    Bx{>��  �          A��ÿ
=q@���A�p�B�{B��
=q���A���B��CsE                                    Bx{>�|  �          A��Ϳ(��@��HA�33B���B��=�(�ÿ��A��\B��HCmn                                    Bx{>�"  �          A��\�   @���A��B�z�B��Ϳ   ��\)A�=qB��qCz�                                    Bx{>��  �          A�ff=#�
@�ffA���B��B�8R=#�
���HA�  B��qC���                                    Bx{>�n  �          A�(�>��R@�33A���B�\)B�
=>��R����A�B�
=C���                                    Bx{>�  �          A�>�  @�A��B�(�B��>�  �޸RA�G�B�=qC�(�                                    Bx{>�  �          A��>k�@��RA��RB��RB��H>k����HA��B���C��                                    Bx{>�`  �          A�33>�ff@���A��HB�33B��=>�ff���
A���B�ǮC��                                    Bx{?	  �          A���>�33@��\A�33B���B��
>�33���A��\B�G�C��                                    Bx{?�  �          A�z�>#�
@�G�A�z�B��{B���>#�
�^�RA�Q�B��=C�@                                     Bx{?&R  �          A�=q>��@�{A��B�B�
=>���:�HA�{B��C���                                    Bx{?4�  �          A��>�p�@�{A�{B��B�� >�p��}p�A��B��C�\)                                    Bx{?C�  �          A���>��@�z�A�ffB�k�B��>�׿�ffA�G�B�C���                                    Bx{?RD  �          A�p�>��
@�33A��RB��B��H>��
��{A���B��RC��3                                    Bx{?`�  �          A�G�>Ǯ@���A��B��B��f>Ǯ��
=A���B��RC�1�                                    Bx{?o�  �          A�
=>��H@�z�A�B�W
B�� >��H��=qA���B���C�/\                                    Bx{?~6  �          A��\>���@�
=A�{B�B�B�W
>����:�HA�ffB�C��                                    Bx{?��  �          A�ff=L��@���A���B�(�B���=L�Ϳٙ�A���B�\)C��                                    Bx{?��  �          A�(���G�@w
=A�\)B��B�녾�G��9��A��\B�
=C��{                                    Bx{?�(  �          A�  ����@q�A�p�B��\B�(������@  A�Q�B��\C�,�                                    Bx{?��  �          A�녽L��@5A�ffB�� B�{�L���|��A�
=B�ǮC��                                     Bx{?�t  �          A�  ?�\@��A�
=B��B�L�?�\�@  A�p�B�\)C��                                    Bx{?�  �          A��?\)@��A�\)B���B��=?\)?^�RA�B�Bd�\                                    Bx{?��            A��>��H@���A��B��B�W
>��H?\)A���B�8RBC
=                                    Bx{?�f  �          A��?!G�@�{A�  B�u�B��?!G�>�A�p�B�  B                                      Bx{@  �          A�33?   @�ffA�=qB��qB��)?   >L��A��B�B�A��R                                    Bx{@�  �          A���?n{@�RA�z�B��qB��R?n{?:�HA���B��Bff                                    Bx{@X  �          A���=u@���A�p�B�B�B�
==u>�\)A�z�B��=B�8R                                    Bx{@-�  �          A���?
=q@�Q�A�(�B���B��)?
=q��A��\B�8RC�Z�                                    Bx{@<�  �          A�ff>L��@θRA��RB�Q�B�\)>L�;�33A�ffB��fC���                                    Bx{@KJ  �          A�{?333@��A�Q�B���B��R?333���RA�B���C��H                                    Bx{@Y�  �          A�p��L��@�33A�\)B�W
B�ff�L�;B�\A�p�B��{C`ٚ                                    Bx{@h�  �          A�33<#�
@�33A�B��3B��
<#�
���A��B��{C���                                    Bx{@w<  �          A��\=L��@��A�{B��B�\=L�Ϳh��A�ffB�=qC���                                    Bx{@��  T          A�=q��=q@�Q�A��HB��B�𤾊=q���A�{B���Ct                                    Bx{@��  �          A�  ����@�=qA��B��B��������B�\A��
B�Cx�H                                    Bx{@�.  �          A��
>�z�@���A���B�{B��q>�z����A�\)B��C�9�                                    Bx{@��  �          A�p�?fff@o\)A��RB���B�  ?fff�I��A��B��
C��R                                    Bx{@�z  �          A��R>�p�@��A�
=B���B��>�p��p�A��B��C�=q                                    Bx{@�   �          A�  >8Q�@�p�A�p�B���B��R>8Q��
=A�G�B�ǮC���                                    Bx{@��  �          A���>�33@�\)A�p�B��qB���>�33��A���B���C��f                                    Bx{@�l  �          A��>u@��A��\B��3B�k�>u��
=A�ffB��C���                                    Bx{@�  �          A��ý��
@��
A�  B���B��f���
��p�A�ffB�ǮC���                                    Bx{A	�  �          A��;8Q�@�Q�A���B�ǮB��
�8Q����A�Q�B�z�C�ٚ                                    Bx{A^  �          A��Ϳ�@�33A��HB��
B��q�����\A�z�B��RCws3                                    Bx{A'  �          A��R�#�
@��HA��RB���B�\�#�
���
A�Q�B�aHCs��                                    Bx{A5�  �          A��R����@�33A���B��HB�8R�������A�ffB���C�]q                                    Bx{ADP  �          A�z�L��@�33A�
=B�B�B���L�ͿǮA�  B��3C��                                    Bx{AR�  �          A�Q�?xQ�@�G�A�G�B�ffB�G�?xQ��\)A�p�B�ǮC���                                    Bx{Aa�  �          A�(�?�{@�A��HB�z�B��q?�{�޸RA�\)B���C�S3                                    Bx{ApB  �          A��?��@���A��B��3B�L�?����33A�G�B���C��                                    Bx{A~�  �          A��>�p�@��A��B��)B�W
>�p���{A�
=B�u�C���                                    Bx{A��  �          A��þ�ff@��A�\)B��{B�k���ff��G�A�z�B��=C}��                                    Bx{A�4  �          A���=���@���A���B���B���=��Ϳ�33A�ffB�z�C�&f                                    Bx{A��  �          A�zὸQ�@�(�A�ffB�W
B��)��Q쿥�A�(�B�C���                                    Bx{A��  �          A�(���G�@���A�=qB�ǮB�(���G���z�A��B�\C|��                                    Bx{A�&  �          A�녾�\)@���A�{B���B���\)��A��B�33C�u�                                    Bx{A��  �          A�    @��A�B���B�{    ����A�\)B��C��                                    Bx{A�r  �          A��
�8Q�@�p�A�(�B�aHB��H�8Q��ffA�\)B��\C��q                                    Bx{A�  �          A���\)@��A��RB�ffB�.�\)��
=A���B��C��                                    Bx{B�  T          A�33��33@�G�A�Q�B�aHB��ᾳ33����A�z�B�B�C���                                    Bx{Bd  �          A�33��\@�A�z�B��B����\��A�Q�B�aHC�)                                    Bx{B 
  �          A�
=�:�H@��HA�z�B�8RB�33�:�H��A�  B��C{h�                                    Bx{B.�  �          A��ÿ(�@��A�ffB�.B�=q�(��
�HA�  B�C~\)                                    Bx{B=V  �          A��Ϳ#�
@�(�A�(�B�
=B��#�
�
=qA��
B�C}h�                                    Bx{BK�  �          A��R�8Q�@��A�  B�BĨ��8Q��Q�A�B��qC{\)                                    Bx{BZ�  �          A�z�
=@��HA��B�8RB�  �
=�p�A��B�� C~�3                                    Bx{BiH  �          A�Q쾏\)@�Q�A�p�B�ffB�ff��\)�33A��B��3C�)                                    Bx{Bw�  �          A�(���@��
A��B�B�������A�33B���C�*=                                    Bx{B��  �          A�  ��{@�33A�\)B�=qB��þ�{��RA���B���C��{                                    Bx{B�:  �          A���@��
A��B�#�B�녽��{A���B���C�ff                                    Bx{B��  �          A���>.{@��
A���B��B��>.{��RA���B�C�"�                                    Bx{B��  T          A�p�?J=q@��RA��\B�=qB���?J=q���A�ffB�z�C�,�                                    Bx{B�,  T          A��?z�H@��
A�ffB�.B�=q?z�H�   A��RB���C�!H                                    Bx{B��  �          A��
?�p�@�ffA��B�{B���?�p���A���B���C�%                                    Bx{B�x  �          A�{?�Q�@�{A��\B���B�8R?�Q��(�A���B�k�C��                                    Bx{B�  �          A�Q�?B�\@�A�  B���B��{?B�\�!�A���B��{C�XR                                    Bx{B��  �          A�(�?�(�@�  A��B��B��f?�(���Q�A�G�B���C��R                                    Bx{C
j  �          A�{@0��@��HA��RB��{B��@0�׿E�A�z�B�Q�C�,�                                    Bx{C  �          A��@�@�33A��RB�p�B��{@녿G�A��\B���C��                                    Bx{C'�  �          A��?xQ�@��RA�  B��3B��
?xQ�� ��A�z�B��qC��                                    Bx{C6\  �          A��?���@��
A�  B�� B���?��ÿ��A�ffB�C��R                                    Bx{CE  �          A��?�Q�@���A�p�B�� B�=q?�Q쿆ffA���B�{C�"�                                    Bx{CS�  �          A�p�?���@ǮA�33B��
B�z�?��ÿxQ�A���B��3C��R                                    Bx{CbN  �          A�G�?ٙ�@�G�A�Q�B�L�B��
?ٙ��(��A���B���C�Z�                                    Bx{Cp�  �          A�
=?��@��
A���B�ffB��)?���Y��A�z�B�W
C���                                    Bx{C�  �          A��R?�G�@�  A���B�\B���?�G��z�HA�(�B��C��                                    Bx{C�@  �          A�z�?�@�
=A�=qB��3B�{?���G�A��B�p�C��)                                    Bx{C��  �          A�(�?�@��A�  B��B��R?�����A�G�B�=qC��\                                    Bx{C��  �          A��
?�(�@�33A�  B�ǮB��f?�(����A��B��C��                                    Bx{C�2  �          A�?�
=@�Q�A�(�B�G�B�8R?�
=��  A�
=B�� C�z�                                    Bx{C��  �          A�  ?�Q�@���A��B�  B��?�Q���
A�G�B�G�C���                                    Bx{C�~  �          A�  ?��\@�=qA�Q�B�8RB�\)?��\��p�A�G�B�8RC��                                    Bx{C�$  �          A�?E�@�(�A�p�B�.B�u�?E����HA��B���C��                                    Bx{C��  �          A�p�?���@ڏ\A���B�\)B�.?��þ�(�A��HB�  C�b�                                    Bx{Dp  �          A�\)?��@���A��
B�  B�?��>#�
A��HB��)@�p�                                    Bx{D  �          A�
=?\@�(�A���B�#�B��q?\>\)A��\B��@�z�                                    Bx{D �  �          A���?�  @���A�Q�B��=B�Q�?�  ��A��\B���C��f                                    Bx{D/b  �          A���?�G�@�=qA�
=B�\)B���?�G���A�=qB�(�C�C�                                    Bx{D>  �          A���?��@�
=A�Q�B�=qB�z�?�녾��RA�{B���C��{                                    Bx{DL�  �          A�z�?�
=@�(�A���B�8RB��\?�
=���A��B���C�q�                                    Bx{D[T  �          A�ff>aG�@�p�A��
B���B���>aG��/\)A���B���C�L�                                    Bx{Di�  �          A�(�>�z�@�ffA�B�u�B�  >�z��.�RA��RB���C��                                    Bx{Dx�  �          A�  >�33@�A��B��B��
>�33�   A��RB��
C�                                      Bx{D�F  �          A���>aG�@��A�z�B�L�B�>aG��
=A�z�B��C���                                    Bx{D��  �          A��>�33@�G�A��\B�aHB��>�33���A�z�B�aHC�q                                    Bx{D��  �          A�p�?#�
@�ffA�  B�B�B��?#�
�\)A�ffB��fC��                                    Bx{D�8  �          A�\)?:�H@�p�A�\)B��B��R?:�H��A�ffB��HC���                                    Bx{D��  �          A�33?u@��RA�{B�G�B�{?u�   A�B��C���                                    Bx{DЄ  T          A���?
=q@�
=A�\)B�#�B�#�?
=q�\)A��
B���C�˅                                    Bx{D�*  T          A���?Y��@��A���B�\B�?Y���
=A��
B�33C��
                                    Bx{D��  �          A��\?L��@�{A��B�(�B���?L�Ϳ�\A��
B�C�+�                                    Bx{D�v  �          A��\?У�@�ffA�  B�u�B�?У�� ��A�33B�aHC���                                    Bx{E  �          A�Q�?�=q@�  A��
B��{B��)?�=q��(�A��B��\C��                                    Bx{E�  �          A�{?�@�G�A���B��B�Ǯ?���z�A��HB���C���                                    Bx{E(h  �          A�{?Ǯ@�(�A�B��B���?Ǯ�
=A��RB��C�5�                                    Bx{E7  �          A��
?���@�\)A�\)B�k�B�Q�?���� ��A���B��C��                                    Bx{EE�  �          A��?��@��A�33B�z�B�?������A���B�u�C�
                                    Bx{ETZ  �          A���?��@��
A���B�
=B�#�?�녿�33A��\B�k�C�q�                                    Bx{Ec   �          A���?��@�=qA��B�aHB��
?����(�A��\B�8RC��                                     Bx{Eq�  �          A�\)>�33@�z�A���B�� B�ff>�33����A��HB��HC�*=                                    Bx{E�L  �          A�\)>�G�@�ffA�{B��\B���>�G��˅A���B���C��{                                    Bx{E��  
�          A�33?Y��@ÅA�p�B�W
B�33?Y����
=A���B��=C�W
                                    Bx{E��  
�          A�33?���@��A��RB���B�=q?����A�  B�B�C�                                      Bx{E�>  �          A�
=?c�
@\A�G�B�p�B�ff?c�
��p�A�ffB�.C�}q                                    Bx{E��  �          A���?O\)@��
A�z�B�B���?O\)��Q�A��\B���C��                                    Bx{EɊ  �          A��R?aG�@ҏ\A��B�k�B�Ǯ?aG��xQ�A�ffB��C�
=                                    Bx{E�0  �          A�ff>�  @���A�(�B��HB��)>�  ���HA�Q�B���C���                                    Bx{E��  �          A�(��\)@أ�A���B�Q�B�� �\)�G�A��B��Cj\                                    Bx{E�|  �          A�{=#�
@�ffA���B��)B�aH=#�
�\(�A��
B�\C�0�                                    Bx{F"  �          A��
>8Q�@���A���B�B��>8Q��\A�B��fC���                                    Bx{F�  �          A�  ?��@�Q�A�
=B�ffB�#�?�녿���A��B�C�Y�                                    Bx{F!n  �          A��?��\@��A��\B��{B��\?��\�n{A��B�{C�޸                                    Bx{F0  �          A�?�33@�ffA��B��B�G�?�33��RA�p�B�� C��)                                    Bx{F>�  �          A��?��@�G�A��B���B��f?�녿O\)A�G�B���C�]q                                    Bx{FM`  �          A��?�Q�@�ffA��B��B��f?�Q�h��A�
=B�ffC�=q                                    Bx{F\  �          A��?У�@���A���B��{B��H?У׿\)A���B�{C��                                     Bx{Fj�  �          A��?�@�z�A�B�� B�z�?���(�A�ffB�=qC�b�                                    Bx{FyR  �          A��\>��H@��RA�  B�#�B�Ǯ>��H>.{A�z�B�A��                                    Bx{F��  �          A�Q�>��@�33A�33B�#�B�\>��>���A�(�B���B�                                    Bx{F��  �          A���  A��A�\)Br\)B�Ǯ��  @G�A�(�B�(�B��
                                    Bx{F�D  �          A�녾uA�A��Bx�B�
=�u?�=qA��B�B�B�=q                                    Bx{F��  �          A��?fff@��A�ffB��B��f?fff>�\)A�\)B��A��R                                    Bx{F  �          A�?�=q@�A�33B��fB�(�?�=q=#�
A�p�B�B�?�ff                                    Bx{F�6  �          A��H?�{@��HA�Q�B���B��\?�{�:�HA�{B���C��f                                    Bx{F��  �          A���?�{@���A��B���B��q?�{���HA��
B�.C���                                    Bx{F�  �          A�ff?���@�\)A�\)B���B�B�?���>L��A�(�B���A)G�                                    Bx{F�(  �          A�Q�?B�\A��A�=qBx��B�p�?B�\?ٙ�A���B�\B��3                                    Bx{G�  T          A��\?z�HA�A��B|33B�L�?z�H?�{A�  B�.BX�                                    Bx{Gt  �          A�z�?k�A
�RA��B~  B��
?k�?�Q�A��B�#�BQ                                      Bx{G)  �          A�ff?�=qA��A�z�B��B���?�=q?��
A��B�B�B.                                    Bx{G7�  �          A�Q�?��A�A�G�B�ffB��H?��?333A�B��fAՅ                                    Bx{GFf  �          A�=q?�G�@�{A��B�B�B�  ?�G��8Q�A��B�\C��                                     Bx{GU  �          A�ff?�z�@�(�A���B���B���?�z�>�\)A��B��=A0Q�                                    Bx{Gc�  �          A��H?�{A�RA�Bn�HB���?�{@A�\)B��Bo�                                    Bx{GrX  �          A�(�?�ffA�
A�33B{Q�B�=q?�ff?��A��B�  BL�\                                    Bx{G��  �          A�\)>�@�z�A�ffB���B�L�>��   A�Q�B���C���                                    Bx{G��  �          A�p�?�@�{A�
=B��B�L�?��P  A�
=B�(�C�                                      Bx{G�J  �          A��H?��?�p�A�p�B�p�B[z�?������A�p�B�� C��=                                    Bx{G��  �          A��H?޸R?\)A��B���A�G�?޸R��\)A�\)B���C��                                    Bx{G��  �          A�G�?��\��G�A���B�aHC��f?��\���A�  B�(�C���                                    Bx{G�<  T          A�33?&ff@�\)A��B��RB��q?&ff>��A�\)B�k�B                                    Bx{G��  �          A��\?0��A��A�z�Bx  B�L�?0��?��
A�  B�
=B��\                                    Bx{G�  �          A�ff?5A�\A�Q�Bl
=B���?5@!G�A���B�k�B��=                                    Bx{G�.  �          A�(�?@  AffA��Bv=qB���?@  ?��A�p�B�=qB�                                      Bx{H�  �          A�?^�RA�RA���Bup�B��?^�R?�
=A���B��Bz33                                    Bx{Hz  �          A�33?fff@�ffA�{B�B�B��R?fff=uA�
=B���@w�                                    Bx{H"   �          A��\?O\)@���A�{B�B�B���?O\)�G�A�Q�B�=qC��)                                    Bx{H0�  �          A���?h��@ʏ\A�=qB��B��R?h�ÿ�
=A�ffB��HC�7
                                    Bx{H?l  �          A�p�?�z�@�33A�(�B��)B�{?�z��?\)A�33B��qC��R                                    Bx{HN  �          A��
?��@�Q�A��B���B���?���W
=A�
=B�Q�C��q                                    Bx{H\�  �          A�{?Tz�@�\)A�{B�{B�?Tz��*�HA�ffB�Q�C���                                    Bx{Hk^  �          A�=q?z�H@�A�Q�B�B�B��\?z�H�p  A�
=B�ǮC�U�                                    Bx{Hz  �          A�  ?�33@Z=qA�33B�k�B�?�33��Q�A�\)B�  C�&f                                    Bx{H��  
�          A��?��@�p�A�  B�#�B��?���>�RA�G�B�ffC�!H                                    Bx{H�P  �          A��?��R@���A�Q�B�B�� ?��R�H��A��B�.C���                                    Bx{H��  �          A��?��H@�
=A���B���B���?��H�o\)A�z�B�k�C���                                    Bx{H��  �          A�?�G�@z�HA�Q�B��3B�8R?�G�����A�{B��fC���                                    Bx{H�B  �          A��?��@I��A�G�B�\B���?����=qA�z�B�33C�9�                                    Bx{H��  �          A�33?��R@+�A�\)B�G�B�k�?��R��Q�A��HB�.C��)                                    Bx{H��  �          A�G�?��H@`��A�{B�\B���?��H��ffA�z�B��C��                                    Bx{H�4  �          A�\)@33@=qA�33B�  BFQ�@33����A��
B�(�C�,�                                    Bx{H��  �          A��R?��@7�A���B��{B�
=?����33A�
=B�aHC��=                                    Bx{I�  �          A�녾�
=@��HA��
B��B����
=����A�33B��3C�q�                                    Bx{I&  �          A����@�(�A��RB�� B���Ϳ�G�A�
=B�G�Cy�                                    Bx{I)�  �          A��
��
=@\A��
B��\B���
=����A��B��C��
                                    Bx{I8r  �          A��
���@�33A�B�k�B��q��녿�A��B��{C��\                                    Bx{IG  
�          A�ff����@���A��\B��B�����Ϳ��HA���B�.C���                                    Bx{IU�  �          A��
��=q@�  A�G�B�� B�
=��=q���HA��B��=C�n                                    Bx{Idd  �          A�\)���@ə�A���B��fB������У�A��RB�u�Cy��                                    Bx{Is
  �          A����J=q@��HA���B��\B�G��J=q��{A��HB��Cs��                                    Bx{I��  �          A�(�=#�
@ָRA�Q�B��B�G�=#�
���
A�B�L�C��\                                    Bx{I�V  �          A�{�B�\@У�A���B��HB�B��B�\���RA���B���C�s3                                    Bx{I��  �          A���!G�@�Q�A���B�B����!G����HA�G�B��qCr}q                                    Bx{I��  �          A�(��+�A   A��B�G�B��=�+�>aG�A��B��3C!�
                                    Bx{I�H  �          A���?�Q�@#33A��B��B�
=?�Q���z�A��\B��C��                                    Bx{I��  �          A�G�@=q�.{A��B�W
C�{@=q��
A�=qBx��C�*=                                    Bx{Iٔ  �          A�G�@.{���RA�33B�B�C��R@.{��A�Bp�HC���                                    Bx{I�:  �          A�@8���A���B��qC�@8����
AyG�Bf{C�t{                                    Bx{I��  
�          A�@E�3�
A��B���C���@E�!��ArffB\p�C���                                    Bx{J�  �          A�p�@Mp��J�HA��HB���C���@Mp��&{An{BWffC��
                                    Bx{J,  �          A�\)@.{�0  A�  B��RC�U�@.{�!�Ar�HB^  C���                                    Bx{J"�  �          A��R@����A�B�B�C��
@���Q�Atz�Bb\)C��                                    Bx{J1x  �          A�  @(��
=A��B���C���@(���Av=qBf(�C�8R                                    Bx{J@  �          A�p�@!G��B�\A��
B��HC���@!G��#�
Alz�BZ  C���                                    Bx{JN�  �          A�p�@������A��B���C���@���4z�A`��BJp�C�                                    Bx{J]j  �          A��@�R�^{A�p�B��
C��=@�R�*{Ah��BTp�C��3                                    Bx{Jl  �          A�\)@���0  A�z�B��HC���@��� ��Ao�B^=qC��                                    Bx{Jz�  �          A�
=@=q?z�A�p�B�=qAW33@=q���HA��
B�#�C�g�                                    Bx{J�\  �          A���@?�{A�p�B���A�p�@���
A���B��)C��H                                    Bx{J�  
�          A���@���ffA��B���C���@�� Q�A�  By��C�=q                                    Bx{J��  �          A�G�@�>�A�  B���@L(�@���A��RB�Q�C��
                                    Bx{J�N  �          A���@&ff���HA���B�  C�U�@&ff��HAx  Bl(�C��                                    Bx{J��  �          A��\@'���p�A�=qB�\)C�z�@'���
Aw33Bf  C��R                                    Bx{JҚ  �          A���@'���  A���B�
=C��@'����A{\)Bk�C��                                    Bx{J�@  �          A�Q�@'����RA�z�B�ǮC�XR@'��A|��Bn�RC�<)                                    Bx{J��  �          A�  @;���z�A��
B��C�1�@;��(�A{�
BnffC�>�                                    Bx{J��  �          A���@)���fffA��
B�W
C���@)���z�A}Brp�C���                                    Bx{K2  �          A�@!G��p��A�(�B�C��{@!G��	p�A}Br=qC�/\                                    Bx{K�  �          A�ff@�H��\)A���B�=qC���@�H���A}BpG�C���                                    Bx{K*~  �          A��H@#�
����A�
=B��C�W
@#�
�(�A|��BmG�C��                                    Bx{K9$  �          A���@*=q��Q�A���B���C��=@*=q���A{
=Bk\)C�'�                                    Bx{KG�  �          A�ff@ �׿�G�A���B�Q�C���@ ���33A|(�Bm�HC��
                                    Bx{KVp  T          A�
=@S33�z�A�ffB��C�\@S33��A|��Br�HC���                                    Bx{Ke  T          A���@Mp��(�A�ffB�33C��f@Mp��(�A|z�BrC���                                    Bx{Ks�  �          A�z�@U��A��B�p�C��@U��A|z�Bs��C�/\                                    Bx{K�b  �          A��
@Vff��A�G�B�=qC��H@Vff�{Az�HBr�
C�5�                                    Bx{K�  �          A���@S�
��33A�
=B���C���@S�
��\)A{�Bt�C�AH                                    Bx{K��  �          A���@p  >�\)A�z�B�(�@���@p  ��  A�G�BzQ�C�H�                                    Bx{K�T  �          A��@tz�>�A��
B��=@��H@tz����A�G�B{�
C��f                                    Bx{K��  �          A��H@l(�>�A���B�=q@��@l(����A�
=B|Q�C�g�                                    Bx{Kˠ  �          A��H@h��>B�\A��
B�@>{@h����=qA�Q�By\)C��
                                    Bx{K�F  �          A��\@Y���#�
A��B�G�C���@Y������A~{Bv33C���                                    Bx{K��  �          A�{@U����A��B���C��@U���\)A|��Bu33C�T{                                    Bx{K��  �          A���@@�׾\A��\B��3C�o\@@���ffA}Bu(�C�!H                                    Bx{L8  �          A���@=p�����A���B�.C��@=p���A�(�Bx��C�=q                                    Bx{L�  �          A���@E��#�
A���B�k�C��f@E����HA�(�By  C��
                                    Bx{L#�  �          A��\@L�;W
=A�Q�B��\C�!H@L��� (�A~{Bv  C��                                    Bx{L2*  �          A�G�@C�
�B�\A�33B��\C�1�@C�
� ��A�Bv�RC�g�                                    Bx{L@�  �          A��R@���z�A�
=B�33C�f@���RAx��Bm33C�t{                                    Bx{LOv  �          A�=q@���HA�{B�z�C�Y�@�(�Ao\)Bfp�C�Y�                                    Bx{L^  �          A��?����A��
B���C���?���p�Aj=qB_
=C�s3                                    Bx{Ll�  �          A�(�@�
���A��\B�u�C�O\@�
�  AtQ�Bmp�C�aH                                    Bx{L{h  �          A��
?����G�A���B��
C�` ?����RAs�Bl�C���                                    Bx{L�  �          A��@�
��\)A�z�B��C�7
@�
�(�Ar=qBj\)C�s3                                    Bx{L��  �          A��
@z῔z�A�=qB�33C��q@z����As33Bl=qC�]q                                    Bx{L�Z  �          A�@33�xQ�A��\B�L�C�\)@33�
�RAu�Bo=qC���                                    Bx{L�   �          A�  @!G����A�=qB�{C��\@!G���As33Bk��C��
                                    Bx{LĦ  �          A��@!G���  A�  B�aHC�1�@!G��
�HAs�Bm33C��                                    Bx{L�L  �          A�=q@5�B�\A�Q�B���C�|)@5��
AuBo
=C�<)                                    Bx{L��  �          A��\@!녿�  A�(�B�{C��@!��{AnffBc��C���                                    Bx{L�  �          A�{@���p�A�B���C�e@���Al  B`C��                                    Bx{L�>  �          A��
@1��#33A��\B�ffC���@1�� z�Ae��BX=qC��                                     Bx{M�  �          A�  @Vff��A�(�B�L�C�^�@Vff�p�Af{BX��C�c�                                    Bx{M�  �          A��H@Dz῵A�{B���C��R@Dz��  Al��Be{C�n                                    Bx{M+0  �          A��@�Ϳ��
A��B��C���@����Aj�HBe�C���                                    Bx{M9�  �          A�G�@(����
A�\)B�W
C�q�@(����Ah��Bb�
C��H                                    Bx{MH|  �          A�
=?�ff�(�A�\)B�G�C���?�ff��RAeB^�HC���                                    Bx{MW"  �          A���?�33�  A���B�u�C�3?�33�\)Adz�B]�C��=                                    Bx{Me�  �          A�@U��(�A�B��=C�O\@U����Ab�RBYp�C���                                    Bx{Mtn  �          A�(�@l(��{A��
B��=C�y�@l(��{Ab�\BW�RC�z�                                    Bx{M�  �          A�@��\�FffA���B�u�C�4{@��\�$(�AV�HBI\)C�q�                                    Bx{M��  
�          A��\@��R�e�A��RB�p�C���@��R�)�AO�BB
=C�p�                                    Bx{M�`  �          A���@���W�A��B�33C���@���'
=ARffBE\)C�5�                                    Bx{M�  �          A�  @Fff�#�
A�{B�C�9�@Fff��A\��BT�\C���                                    Bx{M��  �          A�\)?��R��A���B���C��\?��R���Ad��Ba��C��                                    Bx{M�R  �          A�p�?�zῌ��A���B�Q�C�T{?�z���Al  Bl�C��=                                    Bx{M��  �          A�p�?�\)��p�A��RB��HC�>�?�\)���Aj�HBj�HC��                                     Bx{M�  �          A�p�@=q��p�A��HB�C�Ff@=q��RAa�B]��C�,�                                    Bx{M�D  �          A�
=@Q��
=A�
=B���C��=@Q��=qAb�\B_�C�ff                                    Bx{N�  �          A��H@�
=A���B�  C�g�@���A`��B\C�0�                                    Bx{N�  �          A�
=@��z�A���B��)C�@ @���A_
=BY�HC�'�                                    Bx{N$6  �          A�z�@z��-p�A��B��fC���@z�� z�AZ=qBTffC��
                                    Bx{N2�  �          A�{@#33�[�A��B��fC�P�@#33�(��AQp�BI=qC���                                    Bx{NA�  �          A���?���]p�A��
B�aHC��=?���)��AQG�BJQ�C��=                                    Bx{NP(  T          A��?�
=�Z�HA�z�B��C�q?�
=�)��AR�RBK=qC���                                    Bx{N^�  �          A�G�@33����A}��B�B�C�.@33�BffA9�B,��C�Ǯ                                    Bx{Nmt  �          A�G�@(��#�
A��B�  C��@(���Q�Ak�Bu=qC�޸                                    Bx{N|  �          A��@5��\A�p�B�aHC��@5�p�Ai��BnG�C���                                    Bx{N��  �          A���@�ÿ�A�\)B�Q�C�޸@�����AiG�Bo��C�5�                                    Bx{N�f  �          A�G�@AG���33A��\B�G�C���@AG����AaG�Ba��C�k�                                    Bx{N�  �          A�33@�  ��(�A�Q�B�aHC��f@�  ��AY�BUC��R                                    Bx{N��  �          A�(�@	����(�A�(�B��C���@	����A^{B`  C���                                    Bx{N�X  �          A��@8Q��`��A�z�B�p�C��=@8Q��((�AJffBE
=C���                                    Bx{N��  �          A��?��R�e�A�
=B�� C���?��R�)��AJ�RBF\)C�T{                                    Bx{N�  �          A���@#�
�\)A��B�#�C��q@#�
��HAS�
BS��C�ff                                    Bx{N�J  �          A��H@%�A�{B��fC��=@%���AW�BX�C���                                    Bx{N��  �          A�
=@#�
��
=A�z�B���C�}q@#�
��
AYG�B[(�C��)                                    Bx{O�  �          A�  @
=�A�A���B�#�C�s3@
=�!AMBL�C��f                                    Bx{O<  �          A��\?���(��A�
B�
=C��H?����AO33BRQ�C�z�                                    Bx{O+�  �          A~�\@�H��  AqB��C��\@�H�,(�A6�RB8
=C�U�                                    Bx{O:�  �          A|��@��.{Ax��B��{C�7
@�����AW�Bk{C��f                                    Bx{OI.  �          A|  @;�?�Aw\)B�33A#�
@;���{A_
=ByffC�Ф                                    Bx{OW�  �          A~�H@`  ?O\)Ax(�B��AQ@`  ��
=Aa�By�C�4{                                    Bx{Ofz  �          Azff@�L��Av{B���C��3@���AXz�Bq�RC�޸                                    Bx{Ou   �          Atz�@)���+�Ap  B�aHC�� @)�����AO�Bi=qC���                                    Bx{O��  �          Av�\?aG���33AqG�B��
C�
=?aG���AIp�B_��C��R                                    Bx{O�l  �          Ax��?}p���G�AvffB�
=C��=?}p���AL��B^\)C�=q                                    Bx{O�  �          At��?˅��z�ApQ�B���C��?˅�\)AHQ�B^
=C�S3                                    Bx{O��  �          As�
?����Q�Ao�
B�  C�S3?���33AEBY(�C�Ф                                    Bx{O�^  �          Ar{?��
�AmG�B��{C��)?��
��A@z�BR�HC��R                                    Bx{O�  �          Ar=q@Q����Al��B�=qC���@Q����A?
=BP33C���                                    Bx{O۪  �          Au�?��H�&ffAo�
B��=C���?��H���A@��BOz�C�7
                                    Bx{O�P  �          Au@Q��'
=Ao�B���C��@Q����A@z�BNG�C�t{                                    Bx{O��  �          Aw�
?k��W
=Aq�B���C���?k�� (�A<  BE��C���                                    Bx{P�  �          At  ?��
�3�
An�\B�C�<)?��
�\)A=BL=qC��{                                    Bx{PB  �          AmG�?u���A_�B���C���?u�.=qA ��B*z�C���                                    Bx{P$�  �          A��?����=qAz�\B�ffC��?���.�HA>�HB=�C�|)                                    Bx{P3�  �          A��?���#�
A~ffB�ǮC���?���\)AMp�BQ��C�:�                                    Bx{PB4  �          A���@\)��A���B�p�C��H@\)��p�Ab�HBq�\C��q                                    Bx{PP�  �          A�ff@(Q�?��RA�(�B�A��@(Q���G�Ao
=B�z�C��                                    Bx{P_�  �          A��?�Q�?�  A��RB��qB
�?�Q���G�An=qB�  C�                                      Bx{Pn&  T          A���?��H?�(�A���B�  B4��?��H�ӅAp  B�C�/\                                    Bx{P|�  �          A�p�?�  @p�A��B�L�B_�?�  ��p�Aup�B�#�C�b�                                    Bx{P�r  �          A��@�@C33A�B���BR��@���=qAxz�B��C��                                    Bx{P�  �          A��?��@(�A�(�B��B��?����Q�Ax  B�33C�Q�                                    Bx{P��  �          A�?��\@_\)A��\B�z�B�p�?��\���A}�B��=C�ff                                    Bx{P�d  �          A��
?fff@^{A���B��HB�  ?fff����A}�B�z�C��)                                    Bx{P�
  �          A�?�{@HQ�A�
=B�B�\?�{��33A{33B���C�j=                                    Bx{P԰  �          A�=q?�  @H��A�G�B��B���?�  ���A{�B���C��{                                    Bx{P�V  �          A��׾k�@�Q�A}B��B����k��R�\A��
B�  C���                                    Bx{P��  �          A���.{@���A~�HB���B��q�.{�E�A�\)B��RC�w
                                    Bx{Q �  �          A�33���H@�\)Aw�B�  B��
���H��
A�B��Co��                                    Bx{QH  �          A��R��
=@�  Ax  B�(�B��H��
=�G�A���B�ffCmٚ                                    Bx{Q�  �          A�33����@��\Az=qB�B�B������\)A�\)B�.Cv.                                    Bx{Q,�  �          A��\���H@���AqB��)B�aH���H��z�A�33B���C[s3                                    Bx{Q;:  �          A����
@޸RAo\)B�8RB���
�uA�B�B�CM�                                    Bx{QI�  �          A�G���\)@љ�As\)B��qB��ÿ�\)����A��B���CY�                                    Bx{QX�  �          A�p���p�@�p�As
=B�.B�(���p����A�Q�B��fCY��                                    Bx{Qg,  �          A�����=q@�Q�As\)B�G�B�=q��=q���
A��RB�W
C_�
                                    Bx{Qu�  �          A�G��˅@׮ArffB��)BΊ=�˅���\A�=qB���CZ�=                                    Bx{Q�x  �          A�
=��ff@ڏ\Ap��B��qB�z��ff��33A��B�B�CT�
                                    Bx{Q�  �          A�(��\@���ArffB�BθR�\����A���B�ǮCa�                                    Bx{Q��  �          A��Ϳ�(�@ÅAuG�B�  B�aH��(���A��B�#�Cd0�                                    Bx{Q�j  �          A�33��@޸RAp(�B���B�8R�����
A�{B���CQ�=                                    Bx{Q�  �          A����@�p�An�RB�p�B��Ύ����
A�G�B�=qCYff                                    Bx{QͶ  �          A�
=��G�@��At(�B�p�BǏ\��G��Q�A��B�z�Ct�{                                    Bx{Q�\  T          A�Q�L��@��Aw\)B�p�BøR�L���G�A��HB�G�Cz��                                    Bx{Q�  �          A�G����\A��AdQ�Bk�B������\?333A��HB�z�C��                                    Bx{Q��  �          A��H�
=q@�Ao
=B��B��׿
=q�Q�A�z�B�33Cl��                                    Bx{RN  �          A�  �Q�@�Q�Ak33B~=qB��H�Q녿#�
A��B��qCY�R                                    Bx{R�  �          Ay���@�AXz�Bq{B�𤿫�>aG�Axz�B�C*��                                    Bx{R%�  �          At�Ϳ=p�@�=qA`��B��RB�z�=p���{Ar{B���CuaH                                    Bx{R4@  �          Alz�=�Q�?�Q�Aj=qB��B�p�=�Q����RAV�HB�#�C�n                                    Bx{RB�  �          Ak
=?�G�>\)Ai�B�  @�z�?�G����
AN�\Bv��C�1�                                    Bx{RQ�  �          Al��?��
��(�Ai��B�ǮC��{?��
� ��AC�
B`�C�\)                                    Bx{R`2  �          Ah��@L���ffA`Q�B�p�C�Y�@L�����A4��BL�C�@                                     Bx{Rn�  �          Ag�
@�  ��\)A\Q�B�k�C��H@�  ��A3
=BJ�
C���                                    Bx{R}~  �          AYG�@�=�AS�B��@U�@��ǮA:�RBr��C��                                    Bx{R�$  �          A[\)@Q녿��AL(�B��C�@Q�����A$��BJ�C�n                                    Bx{R��  
�          A^ff@@���z=qAO�B�W
C�Ф@@���ffA  B,�HC���                                    Bx{R�p  �          Ag�
@p  �4z�A[
=B��C��f@p  ���A*�\B=��C�C�                                    Bx{R�  �          AeG�@N{�2�\A[
=B�p�C��@N{�Q�A*�RB@\)C��3                                    Bx{RƼ  �          A`��@/\)�'�AXz�B��qC�"�@/\)���A)BD�C���                                    Bx{R�b  �          A^ff?�{�(��AX(�B�#�C��?�{���A)G�BFp�C��3                                    Bx{R�  �          A^ff?����\AZ�\B���C���?���A0  BQ=qC�8R                                    Bx{R�  �          A_�@1���\AX  B�#�C�9�@1��  A+�BHC�
=                                    Bx{ST  �          AT��@\���!G�AIp�B��C��@\���Q�A�B>�C�O\                                    Bx{S�  �          APz�@J�H��AG�B�.C�aH@J�H��=qA�HBF�C��                                    Bx{S�  T          AH��@?\)�z�HAB{B��C��f@?\)��ffA"�\BX��C�                                    Bx{S-F  
�          A@��@8Q�n{A9�B��
C��@8Q����A�
BX�RC�3                                    Bx{S;�  T          A9���G��xQ�A�\B��C�����G�� ��@��HB�C�l�                                    Bx{SJ�            A333�z�����A��BKffC|�3�z��33@�G�A�
=C��                                    Bx{SY8  �          AAp���\)��z�A�BV�C��{��\)�/
=@��\A��C��\                                    Bx{Sg�  �          AJ�\?�  ���A/�B�C��?�  ��  AG�BY�C��\                                    Bx{Sv�  �          AR=q?�@;�AI�B���Bc=q?��aG�AG�B�W
C�J=                                    Bx{S�*  �          AUG�?��H@G�AL��B��fBg��?��H�\��AK�B�\)C��=                                    Bx{S��  T          AW33?�ff@<��AP  B�B�BjG�?�ff�l��AL��B�aHC���                                    Bx{S�v  T          AT��@>�R@C33AIp�B���B6��@>�R�Z=qAH  B�#�C���                                    Bx{S�  �          AP��@e@�G�A;33B~�BN�@e��
=AG
=B���C�xR                                    Bx{S��  �          AK33@L(�@��RA5��B}�\B_�@L(���{AC�B�Q�C�j=                                    Bx{S�h  �          AG\)@]p�@��A&�\Bd{Bl�H@]p�=L��A>�RB���?E�                                    Bx{S�  �          A3\)@�z�A��?�@<��Bg{@�z�@���@���A��
BU33                                    Bx{S�  �          A3�@�{Az�?#�
@R�\Bs��@�{A�
@�G�A�p�BbQ�                                    Bx{S�Z  �          A2=q@��A�?z�@AG�B��\@��A33@��HA�Q�Bqp�                                    Bx{T	   �          A.ff@�=qA�
?�p�A'�Bd=q@�=q@���@\B�HBGp�                                    Bx{T�  �          A(��@�33A	�?�ffA�
Bcff@�33@�(�@�\)BBG=q                                    Bx{T&L  �          A,��@��\A�?�R@Q�Bq\)@��\@���@��\A�ffB_�
                                    Bx{T4�  �          A&{@8Q�A�\�^{����B�8R@8Q�A�?��
@�(�B�8R                                    Bx{TC�  �          A!��?�@�{��
=�0G�B�z�?�A�R�'��r{B���                                    Bx{TR>  �          A(Q�@�\)A ��@l��A��\B^��@�\)@��@�(�B-{B.�                                    Bx{T`�  �          A4z�@\A��@3�
Ag33B`��@\@��@�33B�
B<�
                                    Bx{To�  �          A4��@��A\)?�{@�z�Bo{@��@��\@�z�A��BX��                                    Bx{T~0  T          A3
=@��A�R>B�\?xQ�Bpz�@��A�@��A�z�Ba��                                    Bx{T��  h          A5p�@��A(�������Bp�\@��A(�@��\A���Be��                                    Bx{T�|  |          AK\)@�G�?�\)A3�B{{A��\@�G�����A)��Bdz�C��q                                    Bx{T�"  T          ATQ�@��RA��@��HB�B9�\@��R@`��A%��BJ�A�                                      Bx{T��  �          A\��@��RA\)@���Aȣ�BG�@��R@��RAffB4  B	��                                    Bx{T�n  T          A]�A  A�@�AB7�HA  @�33@θRA�\)B��                                    Bx{T�  T          AaG�A!�A
=?@  @EB,  A!�@�(�@��A��B�                                    Bx{T�  �          Ac
=AG�A Q�@;�A?33B933AG�@�\@�A�33B                                    Bx{T�`  �          Ad��A��Az�@_\)Ab{B3�HA��@ۅ@���Bz�B��                                    Bx{U  �          Ae�A!�A33@EAG
=B/33A!�@�
=@�p�A���B
Q�                                    Bx{U�  �          Ag\)A�RA�@��A�z�B/A�R@�33ABA�z�                                    Bx{UR  �          Aep�A��A  @�(�A��B&�A��@��A�HBQ�A�                                      Bx{U-�  �          AdQ�A�A\)@�A�G�B2�RA�@��A
=BQ�B =q                                    Bx{U<�  �          AeG�Ap�A(�@�
=A�  B9�Ap�@�\)A�BG�B��                                    Bx{UKD  �          Ac�
A�A��@qG�Au��B6(�A�@�\)A ��B	�B�H                                    Bx{UY�  �          A`��A��A!G�@�33A�Q�BC��A��@�A\)B  B                                      Bx{Uh�  �          Ab=qA�HA!G�@�G�A�p�BA�
A�H@�
=A
ffB
=B��                                    Bx{Uw6  �          Ad��A�RA!G�@��
A�ffB>�
A�R@�p�A�BQ�B=q                                    Bx{U��  �          Ad��A'�
A{@z�A�B'(�A'�
@��@��HA�=qB                                    Bx{U��  T          A_\)@�ffA�@���A��BN��@�ff@�=qA   B9�
B�                                    Bx{U�(  �          A^�\@�
=A�@�AӮBNQ�@�
=@��A"=qB;��B�                                    Bx{U��  �          A^�HA�A  @���A�=qBC�A�@��A=qB%  BG�                                    Bx{U�t  �          A\��@���A33@��RB��BVff@���@���A3�BY�B
=                                    Bx{U�  �          A[\)@��A�@��B�BT�R@��@�ffA/\)BT��B(�                                    Bx{U��  �          A\��@�\)A=q@�A�Q�BJ�R@�\)@���A*�\BJ�\A���                                    Bx{U�f  �          A^=q@��A\)@�B�BG�H@��@�ffA-��BN33A�                                    Bx{U�  �          Ab�R@�=qA��A z�B
�HB>G�@�=q@c�
A3
=BQ�
A�                                    Bx{V	�  �          Af=q@ۅAQ�A�\B=qBX��@ۅ@��HA<(�B]��BQ�                                    Bx{VX  �          AiG�@�{A�\A�\B�BY�@�{@��\A@��B_G�A��                                    Bx{V&�  �          Ai��@�  A�
A�\B�BIff@�  @J�HAD(�Bd�A��                                    Bx{V5�  �          Ahz�@�=qA\)A  BffBG�R@�=q@N{AA��Bb(�A��                                    Bx{VDJ  
�          Ag�
@��
AG�@��A�BT��@��
@�p�A5BP33B�H                                    Bx{VR�  �          Af{@�A-��@���AǮBb{@�@У�A)G�B=�B){                                    Bx{Va�  �          Ad��@�\)A1G�@���A���Bc�@�\)@�G�A!�B4G�B1
=                                    Bx{Vp<  �          Ac\)@��HA-G�@��HA�G�Bf�
@��H@ϮA)p�BAG�B-�                                    Bx{V~�  �          Aa�@�A'�
@�{A���BbG�@�@���A+�
BE�B$
=                                    Bx{V��  �          Ad��@�A#
=@�z�A݅BV��@�@��A,(�BC�\B=q                                    Bx{V�.  �          Af{@�  A!p�@�p�A��B\��@�  @�{A6=qBR\)B33                                    Bx{V��  �          Ad(�@�{A
=A�
B�BP��@�{@b�\A?�
Bc�
A�=q                                    Bx{V�z  �          A^�H@�Q�@�(�A>=qBk�B+��@�Q���
AL(�B��=C��H                                    Bx{V�   �          A\��@�33@��A=p�Bm�HB�H@�33��{AHz�B�Q�C�o\                                    Bx{V��  �          A\  @��
@�  A3
=B[
=B{@��
���AC\)B|{C���                                    Bx{V�l  �          AY�@���@\)A=�Bs(�Bz�@�����AD(�B���C�L�                                    Bx{V�  T          A[�
@ҏ\@���A2�HB[�RB33@ҏ\��(�A=��Bp��C���                                    Bx{W�  �          AW�@��\@�=qA2�HBb\)B#�@��\��  AAB�k�C���                                    Bx{W^  �          AV�R@�33@���A:�\BsBK�@�33��ffAI�B�ffC�.                                    Bx{W   �          AX��@�Q�@�  A5�Bb\)BI��@�Q�   AJ�HB��C�f                                    Bx{W.�  �          AS�@h��@�=qA1�Bf33Bl(�@h�þ�AJ�RB��C��q                                    Bx{W=P  �          AX(�@�=q@��A1B^�BS�@�=q��AJffB�G�C�33                                    Bx{WK�  �          AX  @��@��RA5Be�\B\=q@����{AMG�B��C��\                                    Bx{WZ�  �          AW
=@��@��A%�BJ=qB>�R@��>�ffAB{B�.@��                                    Bx{WiB  �          AU��@�(�@�  AG�B7�\B:p�@�(�?�(�A:�\Bsz�A-G�                                    Bx{Ww�  �          AU�@�@ᙚA�B:�BG�
@�?�A>�RB{�AW�                                    Bx{W��  �          AV=q@���@�ffAB=�HBI�@���?��\A@  B  AE�                                    Bx{W�4  �          AUG�@�@��A\)BABX�@�?��AC
=B��Aq�                                    Bx{W��  T          AT��@�G�@�  A$��BJ�BT�@�G�?^�RADz�B�ǮA��                                    Bx{W��  T          AS\)@�@�G�A�B.{BF��@�?�Q�A7�BrQ�A�                                    Bx{W�&  �          AQp�@�@�ffA��B;�BJ�@�?�z�A;�B}�A_\)                                    Bx{W��  �          AN�H@��@��AB3�BWp�@��@�\A9�B}�RA���                                    Bx{W�r  �          AN=q@��@�\)AQ�B1��BY�@��@��A8Q�B}�\A�{                                    Bx{W�  �          ANff@�G�@�
=A�\B4�Bh��@�G�@�A<(�B�k�A�G�                                    Bx{W��  �          AM�@���@�ffA(�B1�RBc�@���@A9�B�\A�                                    Bx{X
d  �          AL��@��@�=qA�\B6z�BfQ�@��@	��A;
=B�� A�z�                                    Bx{X
  �          ALz�@��
@��
AffB=��B^�\@��
?��A:�HB�� A��                                    Bx{X'�  �          AMp�@\@�=qA��B?��B3��@\?(��A4z�Bv33@�p�                                    Bx{X6V  �          AL(�@���@�Q�A{BD33BB{@���?G�A7\)B�  A (�                                    Bx{XD�  �          AK\)@��H@�Q�A�HBFp�BF{@��H?B�\A8  B��RA�                                    Bx{XS�  �          AJ�R@���@�{A��BL33B@��@���>ǮA8  B�\)@�                                      Bx{XbH  �          AF{@�Q�@�{A�B'�B)�H@�Q�?�33A#33B_��A;33                                    Bx{Xp�  �          ADz�@�\)@�(�A   B#  B%(�@�\)?��HA�BYA<��                                    Bx{X�  �          AA�@��@�z�A��BB
=BH��@��?xQ�A.=qB��fA+�
                                    Bx{X�:  T          A@z�@���@��A��BL=qBKz�@���?�A0Q�B��f@ڏ\                                    Bx{X��  �          A?�@�Q�@�Q�Ap�BT�BV=q@�Q�>�p�A2�HB�.@�                                      Bx{X��  �          A?\)@q�@��A�RB`\)B]@q논�A5p�B��C�˅                                    Bx{X�,  �          A?\)@p  @�{A!�Bf��BX��@p  �\A5�B�L�C�R                                    Bx{X��  �          A9p�@.{@|(�A(Q�B�\B]p�@.{��A1B��C�4{                                    Bx{X�x  �          A9�@x��@�Q�A��B_\)BP��@x�þ#�
A)��B��3C���                                    Bx{X�  �          A:ff@��@���A  B4\)BG(�@��?�p�A$Q�Bw{Ay�                                    Bx{X��  �          A:�\@���@���@��HB(��B5��@���?���Ap�Bf=qAh(�                                    Bx{Yj  T          A;�@ə�@�ff@�\)B$�B2=q@ə�?�AQ�Ba33Al��                                    Bx{Y  �          A7\)@�33@ָR@��BffB5��@�33@+�A�
BNffA�
=                                    Bx{Y �  �          A6ff@�
=@�  @�Q�B�B9
=@�
=@�Az�BY�A���                                    Bx{Y/\  �          A5G�@�(�@˅@�  B%�RBB
=@�(�?�A=qBi\)A��H                                    Bx{Y>  �          A5�@���@��\@��B-=qB+{@���?�\)A�Bd\)A(                                      Bx{YL�  �          A5�@�p�@�=qA ��B533B'��@�p�?@  ABh�R@�
=                                    Bx{Y[N  �          A5�@�@��A ��B5G�B'\)@�?=p�ABh�\@��H                                    Bx{Yi�  �          A4  @�{@�Ap�B@  B.\)@�{>�A��Bs�@�=q                                    Bx{Yx�  �          A3�@�(�@�ff@��B1  B+p�@�(�?z�HA�BgQ�A�
                                    Bx{Y�@  �          A3�
@�{@�\)@�RB��B&��@�{?\AG�BW�RAS�
                                    Bx{Y��  �          A4��@��@���A ��B5�HB;�@��?�z�Ap�Br�AA�                                    Bx{Y��  �          A3�@��@��@�\)B.p�B3��@��?��\A��Bip�AH(�                                    Bx{Y�2  �          A2�\@�{@��R@��
B �B4G�@�{?�G�AB_�A�(�                                    Bx{Y��  �          A2�R@�z�@�@��RB8��B4�H@�z�?c�
Ap�Bqp�A�R                                    Bx{Y�~  �          A5�@�(�@�ffA�B^G�B<�@�(���
=A'�B�p�C�:�                                    Bx{Y�$  �          A5p�@�G�@��A�\BW�B<��@�G��\)A&=qB�C�)                                    Bx{Y��  �          A4z�@�(�@�p�A�\BYp�B5  @�(���33A$z�B�k�C��3                                    Bx{Y�p  
f          A4  @��
@�\)A�B\\)B0z�@��
�
=qA#�
B�#�C���                                    Bx{Z  �          A3�@�  @��HA��B`ffB/@�  �333A$  B�
=C���                                    Bx{Z�  T          A2�\@��\@���A�RB](�B/Q�@��\�z�A"�\B�{C�`                                     Bx{Z(b  �          A.�H@���@�G�A��B_�B0��@��Ϳ(�A�B�  C��                                    Bx{Z7  @          A&�R@mp�@��A{Bh��B>=q@mp��=p�A�
B��=C�`                                     Bx{ZE�  �          A&ff@c33@r�\A  Bp�B;Q�@c33���
A\)B�L�C���                                    Bx{ZTT  �          A&ff@l(�@y��A�\Bk��B:{@l(��c�
A�HB�#�C�>�                                    Bx{Zb�  
�          A%@|��@�ffA	��B`��B:�
@|�;��A��B�C��{                                    Bx{Zq�  "          A"�R@�z�@�G�A�B]p�B1\)@�z�   Az�B���C��\                                    Bx{Z�F  
�          A!�@�p�@�(�@�BD�RB3{@�p�>\AffBy  @��R                                    Bx{Z��  
�          A ��@���@���@�z�B-�B9��@���?��A	p�BkG�Ah��                                    Bx{Z��  
�          A�@��
@�@�ffB#\)B:��@��
?��AQ�Bc��A�z�                                    Bx{Z�8  �          A��@��@��@�RB<G�B@{@��?fffA�ByG�A4z�                                    Bx{Z��  T          A=q@�  @��@��BP��BC�@�  >uA��B�8R@Y��                                    Bx{ZɄ  �          Aff@�33@��\@��RBM��B@��@�33>���A\)B��@�                                      Bx{Z�*  �          A (�@��@��H@�Q�B9�RB>{@��?xQ�A��Bvp�A<                                      Bx{Z��  �          A@�(�@��
@��
B)�B?�@�(�?��
A=qBj��A���                                    Bx{Z�v  �          A�@��
@�@�p�B%z�B@z�@��
?�A�
Bh{A���                                    Bx{[  T          A�R@�  @��@�ffB7�RB?�@�  ?��A�
Bu�
AP(�                                    Bx{[�  "          A\)@�{@��@��B*��BI\)@�{?�\)A33Bp�A�                                      Bx{[!h  �          Aff@�=q@��
@�Q�BJ=qB<
=@�=q>�{A�B�u�@��\                                    Bx{[0  H          A�H@_\)@tz�@�33Bb\)B>=q@_\)��
=A�B�\C��
                                    Bx{[>�  -          A(�@*�H@U�A�B}{BM�@*�H��33AQ�B�W
C�^�                                    Bx{[MZ  �          A(�@z�H@g
=@�ffBY�B*p�@z�H���A(�B�  C��                                    
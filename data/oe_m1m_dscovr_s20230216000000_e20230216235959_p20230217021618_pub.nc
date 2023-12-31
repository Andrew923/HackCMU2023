CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230216000000_e20230216235959_p20230217021618_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-17T02:16:18.450Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-16T00:00:00.000Z   time_coverage_end         2023-02-16T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxexU   
�          A^{?��A ��A�\B+33B�Q�?��@��AR{B�
=B�.                                    Bxexc�  
�          A\��?(�A\)Ap�B8G�B��?(�@[�AS\)B�u�B���                                    BxexrL  �          AH  >���@�Q�A5p�B�.B���>��Ϳ�
=AEp�B�p�C�g�                                    Bxex��  
F          AHQ쿕��A@Q�B��=Cp������ffA�BJG�C���                                    Bxex��  |          AL(����H��p�AC�
B��HC]Y����H��  AG�BO  CT{                                    Bxex�>  
�          AL�Ϳ�@@��AD��B���Bޣ׿��Y��AC\)B�.Cz�3                                    Bxex��  �          AN�R���R?�{AL  B��
B�����R���\A>�\B��{C��                                    Bxex��  
�          AR{����?У�ANffB��CY�������z�A@��B���C{��                                    Bxex�0  �          AS33����@&ffAL��B��B�녿����~�RAF�HB�33Cup�                                    Bxex��  	�          AT(���Q�@W
=AJ�HB���B���Q��P��AK33B��RCo@                                     Bxex�|  T          AU����@W�AL��B��RB������S�
AL��B�#�CqE                                    Bxex�"  
�          AW����
@eAMB�p�B����
�I��AO�B��
Cp��                                    Bxey�  
�          AY녿�p�@fffAPQ�B��B�B���p��N{AQ�B��HCq�\                                    Bxeyn  �          A[\)��ff@]p�AR=qB���B��ÿ�ff�X��AR�\B�p�Cr{                                    Bxey"  
�          AX  �+�=�Q�AR�\B���C2+��+���33A8z�Bl�\Cw�                                    Bxey0�  �          AW��^�R��ffAO33B���CH� �^�R��ffA*�RBQz�Ct��                                    Bxey?`  �          AV{��  �<(�AEG�B��{CV����  �	G�Ap�B133Cs��                                    BxeyN  
�          AW33��\)�mp�ABffB��HC[����\)��\A��B#=qCs�3                                    Bxey\�  "          Ad����ff�w
=AL  B|Q�CX����ff�p�A(�B!Q�Cq��                                    BxeykR  
�          Ac\)���
�c33AMB�CX����
�A  B(  Cr�                                     Bxeyy�  �          Ac
=����:�HAR=qB���CT�q����33A ��B4�CsT{                                    Bxey��  "          Af�R��G��4z�AV{B�p�CS�\��G���A$��B6�HCs&f                                    Bxey�D  �          Ah�����H�0��AW33B�
=CQ����H�33A&�\B6��Cq�H                                    Bxey��  
�          AiG���{�FffAW33B���CUs3��{�  A#�
B2��Cs)                                    Bxey��  �          Ahz���ff�:�HAV�HB�33CS�
��ff�G�A$��B5  Cr�H                                    Bxey�6  
�          Ad  ��\)�:�HAS\)B��CU!H��\)��A!�B5
=Csz�                                    Bxey��  T          Ad(�������AUG�B�\)CN�3������A(��B>�RCq�)                                    Bxey��  �          Aa���(���\)AS
=B�33CGG���(���=qA+�
BG
=CoaH                                    Bxey�(  T          Aa���׿�ffAV=qB��{CL
�����G�A-p�BH�HCs�                                    Bxey��  T          A^�\��33���\AR=qB��CD@ ��33��A.{BM�HCo�{                                    Bxezt  
Z          A[\)��ff�L��AMp�B�8RC=����ff��A-p�BQG�Ck�\                                    Bxez  �          A[33��
=>��RAJffB�{C0�
��
=���A3\)B\
=CdO\                                    Bxez)�  T          A`����
=>�  AR�RB��\C1&f��
=��p�A:=qB_
=Cg)                                    Bxez8f  "          Ah���0  ���
Ad(�B�  C:�
�0  ��{AE�Bh
=Cy
=                                    BxezG  T          Ac����R?��AW�B�\C,�����R��G�A@��Bh=qCi��                                    BxezU�  "          AaG���{>\AV�RB�33C.�=��{���A>�HBh
=Ck��                                    BxezdX  �          A_�
��ff?���AL��B�33C$h���ff��p�A>ffBk�
C_aH                                    Bxezr�  �          A_����
@33AJ�HB��C�
���
���
AC\)Bs��CY�                                     Bxez��  T          AX(���Q�@>�RA=�Bu  C����Q��G
=A<��Bs�
CPW
                                    Bxez�J            Ac�
��  @   AO33B���C����  ��=qAH��BxQ�CY�q                                    Bxez��  �          AmG���\)@b�\AVffB��{C����\)�W�AW33B�W
CT�)                                    Bxez��  
�          Apz���(�@qG�AY��B���C�f��(��P��A[�
B�B�CTs3                                    Bxez�<  "          Aq���H@c�
A]B�#�C�����H�dz�A]B��CXk�                                    Bxez��  �          Aq����@W
=A`(�B�\)C0�����s�
A^ffB�
=C\W
                                    Bxezو  T          Arff�w�@1G�Af{B�\)CaH�w���  A^�HB��
CeL�                                    Bxez�.  �          Atz��\��@�Aj�HB��C��\����{A`��B��=Ck
                                    Bxez��  T          Au��J=q?�33An�RB�=qC��J=q����A_�
B�{Cp5�                                    Bxe{z  "          Aw��33?=p�Atz�B�
=C"B��33�ָRA\  Bz�
C{�                                    Bxe{   "          Ax(��z�?�Aup�B�.C%޸�z���p�A[33BxC}^�                                    Bxe{"�  "          Ay���33>�33Av{B��C)}q��33���HAZ�\Bw  C~��                                    Bxe{1l  �          Ay��`��@(�Apz�B�B�C8R�`�����\Ae��B�k�Ckc�                                    Bxe{@  �          Az=q�a�?��Aqp�B�Q�C�3�a���33Ab{B��
Cm�=                                    Bxe{N�  �          Aw33�Dz�?�ffAp��B�  C!H�Dz��\A\��B~��Cs5�                                    Bxe{]^  
�          Ax���tz�@2�\Alz�B�z�C�)�tz���z�Ad��B�ffCf�                                    Bxe{l  "          A|���Z�H@�HAs�B�33C���Z�H��p�AhQ�B�Cl��                                    Bxe{z�  
�          Ay�|��@(Q�AnffB��3CW
�|�����\AeG�B���Cf�                                    Bxe{�P  
Z          Aw\)�O\)?��Ao�B��HCǮ�O\)����A`��B�
=Co�\                                    Bxe{��  "          A~{�Q�ٙ�Ay�B��qCW�f�Q��33AM��BX�C!H                                    Bxe{��  
�          A�33�,(��.{A{
=B���CaQ��,(��ffAG�BJQ�C~�=                                    Bxe{�B  
�          A�=q�E���HAz{B�W
CM^��E�(�APz�BZ{Cz�
                                    Bxe{��  �          A��R�e�?Q�A~=qB���C'��e����
Ae��Bv�Cr}q                                    Bxe{Ҏ  "          A���i��?���A33B�\)CT{�i����33Ak\)B~(�Cp!H                                    Bxe{�4  
�          A�G���z�@C33Aqp�B��C�{��z�����Ak�B\)CZ��                                    Bxe{��  
�          A�\)����@ffA{�B���C.�������AmG�B�RCgW
                                    Bxe{��  �          A�
=�Z=q?�=qA��B��\C�q�Z=q���Am�B~Q�Cr�=                                    Bxe|&  T          A��Z�H?�A|  B���C��Z�H����AjffB�Q�Cpn                                    Bxe|�  T          A�z��c�
@�AxQ�B��C.�c�
����Aj�\B��RCm8R                                    Bxe|*r  �          A}��k�@%�Ar�\B���C�R�k���{Ah��B��Ci\)                                    Bxe|9  
�          A~�R���@s�
AnffB��qC	&f����n�RAn�RB�.C^5�                                    Bxe|G�  
Z          A33�p  @n�RApz�B�33C#��p  �w
=Ap  B��Ca�\                                    Bxe|Vd  �          A~ff�e�@l(�Ap��B�8RC&f�e��x��Ao�
B��CcaH                                    Bxe|e
  �          A}���qG�@�z�Aj�HB���C{�qG��8Q�Aq�B�(�CYc�                                    Bxe|s�  �          A}��Vff@u�Ao�B�k�C&f�Vff�n{Ao�
B�\Cd�                                    Bxe|�V  �          A}��5@%�Au��B��C	���5��  Ak�
B�\)Cpn                                    Bxe|��  
(          A|z�� ��?�\)Ax��B���C	�� ����
=AiG�B��
Cz�H                                    Bxe|��  �          Av=q��{?�G�As�B�33C녿�{���HA`  B�\C}�                                    Bxe|�H  �          Ajff����?\(�Ag�
B�B�C&f������33AR�RB�.C|E                                    Bxe|��  
�          Ap���G�?uAm��B�(�C��G���p�AXz�B�(�Cy�{                                    Bxe|˔  �          Au��ff?
=Ar�\B���C$c��ff����AZ=qB{�RC|s3                                    Bxe|�:  �          At(���ff>��HAr=qB��C$�R��ff�ָRAYG�B{�\C~�R                                    Bxe|��  �          Au����?@  As�B�#�C�R����У�A\z�B�#�C�W
                                    Bxe|��  T          Av{���?:�HAt  B��C�)����љ�A\��B~�
C}��                                    Bxe},  T          Aw\)��\?.{Aup�B���C��\��(�A]��B~�\C
                                    Bxe}�  �          Av{����>�Q�At(�B�#�C(��������
AZ{By�
C&f                                    Bxe}#x  �          Atz��33?��ArffB�{C#�f��33��z�AZ{B|p�C~                                      Bxe}2  
�          Aw
=��?L��At��B�{C� ���ϮA^=qB�.C~.                                    Bxe}@�  "          Ap�Ϳ�녿У�An{B���Ce�
�����HAE�B]z�C�Q�                                    Bxe}Oj  "          Ap(����
��
=AnffB���CCff���
���
AO\)Bop�C�                                      Bxe}^  T          Aq녿��H�L��Ap(�B���C5Ǯ���H���HAS�
Bt�\C�4{                                    Bxe}l�  �          Ar{��z�?�RAo�
B�C!���z���\)AX��B}��C}��                                    Bxe}{\  T          Au���
=?h��Ar�RB���C�3��
=�ə�A]G�B�\C|�R                                    Bxe}�  �          A�(���Q�?(�A���B���C"�Ϳ�Q����
Al  B}�C~�R                                    Bxe}��  �          A����Q�>�\)A��RB�\C*����Q���z�Ai�Bz
=C��3                                    Bxe}�N  T          A�  �G
=@{A�{B��C���G
=����At��B�G�Co��                                    Bxe}��  T          A����33>�{A�Q�B�=qC*�R�33��{Aep�By�
C~{                                    Bxe}Ě  
�          A{�
�녿:�HAxz�B�.CGǮ����z�AVffBjC��                                    Bxe}�@  
�          A}��
=����AyB�ǮCT���
=��AR�HBa�C�                                    Bxe}��  D          A}�
=q��Q�AyB�\CZ��
=q�Q�APz�B]{C��                                    Bxe}��  T          A~=q���ÿ�=qA{�B��Ca������Q�AV�RBh�C�L�                                    Bxe}�2  
�          A}G��8Q쿙��A|Q�B�.C��=�8Q��ffAVffBg�C�c�                                    Bxe~�  �          A{
=���H�G�Ax(�B�Co���H��
AL��BZ�C�*=                                    Bxe~~  �          Aw�
���5As�B��{C�8R���G�AB�\BN��C���                                    Bxe~+$            Az�H���^{Ar{B�  Crz���!p�A<��BD  C��                                    Bxe~9�  
�          A{
=�$z�\At��B��fCR}q�$z��
=AMB^33C}�                                    Bxe~Hp  T          Az�H�Dz�>.{AuB�\)C0ٚ�Dz���
=A[33BsCv5�                                    Bxe~W  2          A}��aG�?�G�Av=qB�8RC#���aG���
=AaB|�Cp}q                                    Bxe~e�  �          A|z��Z=q>�AvffB��
C,J=�Z=q�ָRA]�Bv{Cs
                                    Bxe~tb  "          A~{�p  ?z�HAv=qB��RC%^��p  �ǮAa��BzCn�q                                    Bxe~�  �          A�{�Z=q?��Ayp�B�  C!u��Z=q��AeBG�Cq�                                    Bxe~��  T          A��H�Y��?333A{�B�{C(L��Y�����
Adz�By��Cr��                                    Bxe~�T  "          A���W�?}p�A{�
B��
C#���W����
Af�HB}�
Cr!H                                    Bxe~��  T          A���*=q?���A}p�B�#�C���*=q��\)AiB�G�Cv�H                                    Bxe~��  
�          A����h��?h��Ay�B�  C%�3�h����(�Ad��B{=qCpJ=                                    Bxe~�F            A~�\���R?�=qAt(�B�C"s3���R���\Ab�RB|ffCj33                                    Bxe~��  �          Ay�����@
�HAf�HB�  C&f�����ffA]�Bz\)C\�3                                    Bxe~�  d          Az{��p�@.{Ag
=B��C=q��p����RAaG�B�W
C[�                                    Bxe~�8  
�          A|z��tz�@33Ar�\B�\C�R�tz����
Af�\B�8RCiE                                    Bxe�  "          Aup��5?�p�Ao�
B��HC�{�5����A`  B�  Cr޸                                    Bxe�  
�          At���5?���AmG�B�W
CW
�5���A`Q�B�k�Cq#�                                    Bxe$*  �          As�
�z�H@s33Ac�B�  Cٚ�z�H�G�Af=qB���CZ�                                     Bxe2�  �          Ax����{@~�RAep�B���C
&f��{�@  Ai�B���CV�                                    BxeAv  	�          A����ff@~{Ap��B���C
L���ff�UAs\)B���CXٚ                                    BxeP  �          Au�Mp�?�ffAn�\B�L�C�q�Mp���{Aap�B�(�CnO\                                    Bxe^�  �          Aw33�Tz�@�An�\B�(�C��Tz����HAdQ�B�B�Ck�{                                    Bxemh  �          A|  ��?��Aw
=B��
C������33AiB�Q�Cv
                                    Bxe|  �          Axz��.{?�Ar�HB��{C�f�.{��  AeB�33Cr�H                                    Bxe��  �          Azff�?\)?��
At  B�G�CE�?\)���HAfffB���Cp�R                                    Bxe�Z  �          AuG��B�\@�RAn{B��=C� �B�\��Q�Adz�B���Cmc�                                    Bxe�   �          Ar�H��z�@�ffAP(�Bm��Ck���z�
=Ad��B�  C:�                                    Bxe��  T          Ai����@�
=A;\)BUB�.��?k�AYp�B��RC)�                                    Bxe�L  "          Afff��G�@�\)A4Q�BN{B�W
��G�?�{AU�B�p�C%��                                    Bxe��  "          Aep�����@��AEG�Br��CaH���ͿJ=qAV�RB�.C>&f                                    Bxe�  �          Ak���=qA��A2ffBF�B�����=q@Q�A[
=B�C}q                                    Bxe�>  
�          Al  ���@�z�A5p�BI\)C����?O\)AQB{��C-
=                                    Bxe��  T          Aep����@��RA5�BR��C�
���>W
=AMp�BQ�C2!H                                    Bxe��  T          Adz��љ�@�A(��B?��Cc��љ�?�Q�AH��Bwz�C'�{                                    Bxe�0  
�          AdQ���Q�@���A"=qB7z�Ck���Q�?�=qADQ�Bq�C$޸                                    Bxe�+�  "          A`����33A�HA (�BG�C���33@|(�A/
=BM
=CT{                                    Bxe�:|  
Z          A`������A�@�RA��
C  ����@�G�A'\)BCQ�Cs3                                    Bxe�I"  "          A\Q��ə�A�RA=qB�
B�� �ə�@��\A4z�B_p�C��                                    Bxe�W�  �          AR�\��p�A$��@w�A��B��R��p�@�
=A ��B��C�\                                    Bxe�fn  @          AP���(�A��@�\)A���C� �(�@�{@�
=BQ�C.                                    Bxe�u  �          APQ���=qA�@���A�z�Cٚ��=q@�p�A�B�C�)                                    Bxe���  |          ARff���
A-G�?��@�
=B������
A\)@ə�A��C s3                                    Bxe��`  r          AN�\��  A%?��A33B�����  A
=@ə�A�  C��                                    Bxe��  h          AR=q���\A(  �h���\)B�p����\A��@vffA�z�C ��                                    Bxe���  
P          AR�H�\A3��#�
�5G�B��
�\A333@,��A?\)B���                                    Bxe��R  h          AV=q��p�A<z��-p��;33B�ff��p�A<(�@333AA�B�z�                                    Bxe���            AS\)���A$���]p���B�q���A,(�?�(�@���B��                                    Bxe�۞  �          A\���i��A$����\��B��i��AIp��\)�33B�L�                                    Bxe��D  �          AYG����HADz�>�Q�?�ffB������HA-��@�Q�AˮB�\)                                    Bxe���  �          AR�R���A8(�<��
=���B�  ���A$��@��A�Q�B�=q                                    Bxe��  T          AF�R�0  @����33�M
=B�W
�0  A-�����хBЀ                                     Bxe�6  
�          AJff�7�@����F=qB����7�A2=q��  ��Q�B��f                                    Bxe�$�  
�          AI��<��@�(���@(�B��<��A3
=�����\Bр                                     Bxe�3�  "          AJff�=p�A�H��H�:�B�Ǯ�=p�A6{��z���ffB�#�                                    Bxe�B(  �          AM��@��A���(��9z�B��H�@��A8z���p���\)B�G�                                    Bxe�P�  �          AMG��EA
=��\�0  B���EA;���{���\Bр                                     Bxe�_t  
�          AF�H�8Q�A ���33�933B�aH�8Q�A2�\��Q����B���                                    Bxe�n  �          AG33�z�A Q��\)�?33B�B��z�A3�
������ffB�Q�                                    Bxe�|�  
�          A^=q��p�@陚�;��fp�B�  ��p�A;
=�����B�z�                                    Bxe��f  T          Ab{�ٙ�@��
�>�\�fG�B��ÿٙ�A=G���� G�B�W
                                    Bxe��  �          AdQ���@�
=�Ep��o�BΊ=���A:�\���	�B�\                                    Bxe���  
�          Ar�H�ٙ�@�  �Q��n{B͊=�ٙ�AG
=��
�G�BÏ\                                    Bxe��X  �          Ay녿��
A��O�
�a�B˽q���
AT���   ����B�G�                                    Bxe���  �          A~{��{A=q�X���j  B�ff��{AS��
�\�(�B��f                                    Bxe�Ԥ  �          Anff��G�A ���1���KB�G���G�AA���  ��z�B�                                      Bxe��J  T          Ab=q��Q�A;�@`  Ah(�B����Q�A��A�
Bp�B��                                     Bxe���  �          AhQ����A@  ?��H@���B�Ǯ���A�@޸RA�C p�                                    Bxe� �  
�          AaG���
=AE�\)�
=B�\��
=A333@��RA�{B��f                                    Bxe�<  �          Aep�����APzῨ������B�����AE�@�{A���B�.                                    Bxe��  
(          AiG���=qAZff�\)��B����=qATz�@w�Ax  B�
=                                    Bxe�,�  
Z          Ao
=�H��A]���=q����B͏\�H��Aep�@�
@�\)B̮                                    Bxe�;.  T          As
=��=qAS���=q�z{B�L���=qA[
=@G�@�p�B���                                    Bxe�I�  T          Av�\��G�AR�R������B�L���G�Ab=q?n{@\��B�B�                                    Bxe�Xz  �          A��\��  AVff�����
=B�R��  Ak\)>�\)?�  B�                                    Bxe�g   �          A�������A^ff�\����B������Ar�\>�?�\)B�ff                                    Bxe�u�  �          A~ff��
=AY������(�B�\��
=AmG�?   ?�ffB�.                                    Bxe��l  
�          A����A^�H��
=���B������Ar=q?��?���B݅                                    Bxe��  
�          A�
=���Ad����  ��Q�B�
=���A{�>k�?O\)B٣�                                    Bxe���  T          A����Ai��θR��Q�B�\)��A~�H>�p�?�G�B�\)                                    Bxe��^  T          A�p�����Ac��ə���ffBܞ�����Ax��>�Q�?�  B�ff                                    Bxe��  T          A����33AF�R�G����\B�.��33Aj�H����\)B���                                    Bxe�ͪ  "          A�����AS������B�ff����Au��\)��z�B�aH                                    Bxe��P  
�          Ay��.{A&�\�!G��,z�B�G��.{A[33���R��
=B�u�                                    Bxe���  3          Ae녽�Q�@�{�QG��B���Q�A.�R�G��"
=B���                                    Bxe���  u          Aa���aG�@����K�
ffB�G��aG�A-p��  �B�aH                                    Bxe�B  T          Aap��(��@�(��?��i
=B�=q�(��A<  ����Q�B�p�                                    Bxe��  �          A`(���@�Q��<(��dp�Bţ׿�A<Q����\)B�Q�                                    Bxe�%�  �          Ad(���  A���6�H�U{B˳3��  AF=q��33�噚B��                                    Bxe�44  �          Am���H@�33�HQ��e�B�����HAF=q� ���z�B�u�                                    Bxe�B�  
Z          Af�R��Q�@�{�C
=�f�
B�k���Q�AA�����\��\B�u�                                    Bxe�Q�  �          Af{��A�R�0���IffB�.��AK�
��G����Bƅ                                    Bxe�`&  T          Aj�\�k�A z��:�H�S�B�L��k�AB�H��  ��G�Bգ�                                    Bxe�n�  �          Aap��j�H@�\�4z��U{B��j�HA9����\��p�B��                                    Bxe�}r  �          Ac\)�xQ�A33�$���:ffB��H�xQ�AF�H��(���
=B֣�                                    Bxe��  @          AX����
=A (��������
B����
=A/�
=�\)>��
B�                                      Bxe���  �          ALz����A  �p����  C�����A(�@Mp�AiC33                                    Bxe��d  �          AL  ��A���33�ǮC � ��A\)@u�A��C��                                    Bxe��
  �          ALQ�� Q�Aff?(�@.�RC�� Q�A  @��A��C}q                                    Bxe�ư  �          AO��33A�?�G�@��CxR�33@�=q@�z�A�p�C�=                                    Bxe��V  �          AQG��
�\A ��@���A���C	&f�
�\@���A�B�C��                                    Bxe���  T          AS\)��\)A�R@�Q�A��RB�����\)@�z�@���B��CW
                                    Bxe��  �          AV�H��ffA;
=?E�@S�
B�Ǯ��ffA%G�@���A�p�B���                                    Bxe�H            AYp���33A=�?У�@�ffB�z���33A!p�@˅A�RB�Q�                                    Bxe��  "          AV�H���A�@�33A���C�����@���A\)BffCT{                                    Bxe��  T          ATz����A	�@�=qA��C� ���@�A=qB�Ck�                                    Bxe�-:            A\�����RA-녿����{B�p����RA)p�@>{AI�B��f                                    Bxe�;�  
�          A^�\����A:=q�k��u��B� ����AB{?�z�@��\B�                                     Bxe�J�  �          Ab=q���HA>=q�W
=�\Q�B��)���HAC�?�ff@�=qB�                                     Bxe�Y,  �          AX�����A0����=q��{B�#����A<(�?L��@Y��B�{                                    Bxe�g�  T          AXQ��˅A1p���  ��33B��˅A?�>�
=?�G�B�                                      Bxe�vx  �          At�����
AG�������B��H���
A\�;��R��B�=q                                    Bxe��  �          Am����=qABff������
B����=qAW
=������33B�\)                                    Bxe���  �          Ac�
�ƸRA7\)��Q�����B��H�ƸRAL�ÿ����B�3                                    Bxe��j  �          Adz�����A1����H����B�\����AK
=��ff��\)B�                                    Bxe��  
�          Aa����{A.=q����Q�B�8R��{AJ�R�ٙ����B�B�                                    Bxe���  "          AS
=��  A�R��p���33B�����  A>{�z��!�B�R                                    Bxe��\  �          A\�����HA$(�����
z�B�Ǯ���HAIG��E�Mp�B�\                                    Bxe��  "          Ah������A(Q�����ffB�Q�����AP  �Y���X��B���                                    Bxe��  "          Am��p�A,(��	G��B�{��p�AT���aG��[
=B�.                                    Bxe��N  �          Ah����G�A-G��ff�G�B����G�AS
=�G
=�E��B߸R                                    Bxe��  "          Aj�\���A-p���
��B�����AS��L���I�B��                                     Bxe��  �          Af�\��ffA{���$ffB�G���ffAF=q��p����B�z�                                    Bxe�&@  �          Ahz����
A(Q��
�H�G�B������
AQ�p  �n�\B޳3                                    Bxe�4�  3          Aa����A%���p��
=B����AM��`���f�RB��                                    Bxe�C�  
�          Ay����Az��2�\�7ffB�����AV�R���H��ffB�L�                                    Bxe�R2  T          Aj�H�hQ�A33�!���1
=B�  �hQ�AN�R������\)B�aH                                    Bxe�`�  
�          AY��+�A���)���L�Bأ��+�A:ff��Q����B��                                    Bxe�o~  u          AYp��-p�@�33�.�H�V=qB�Q��-p�A4��������HB���                                    Bxe�~$  �          AR=q�.�R@��H�"�\�J��B�aH�.�RA3���
=��z�B�W
                                    Bxe���  T          AR�R�<(�@�  �&�\�P��B����<(�A0(����H��z�B��                                    Bxe��p  T          AP���B�\@��
�"ff�L(�Bߏ\�B�\A0  ��=q�癚B��                                    Bxe��  "          AQp��+�@�p��,  �]�B�Q��+�A)��������
B�aH                                    Bxe���  �          AQ��	��@�33�3��lz�B�Q��	��A$(������
Bˣ�                                    Bxe��b  �          AQ��G�@�
=�4  �k�RBֳ3�G�A%������
B�\                                    Bxe��  �          AU��33@����4���j(�B��f�33A'
=��33��B��f                                    Bxe��  �          AT�Ϳ���@��
�.�R�\�RB��Ϳ���A0������Q�Bƽq                                    Bxe��T  u          AUp����R@�Q��8���p��BɊ=���RA(  �G����B�p�                                    Bxe��  �          AUG��'�A z��$���J�B���'�A6�\��33��B��)                                    Bxe��  �          AY녿��R@�(��J�H
=B�Ǯ���RA33�{�:��B�p�                                    Bxe�F  �          A�z��W�A
=�U��\z�B�\)�W�AS33�
=�
=BО�                                    Bxe�-�  �          A����z�A ���C�
�A��B�Ǯ��z�A`(���(���z�B��                                    Bxe�<�  �          A�����RA+\)�:�R�4�RB�.���RAfff�Ӆ���B�ff                                    Bxe�K8  �          A�  ��33A8���/\)�#��B����33Anff��z�����B���                                    Bxe�Y�  �          A���QG�A#33�P���Jp�B׏\�QG�Ag
=�p���33Bͅ                                    Bxe�h�  �          A���L��A!��S�
�M�B�
=�L��Ag
=�����RB��                                    Bxe�w*  �          A�=q��  A1��@���4��B���  Anff��33��  BՔ{                                    Bxe���  �          A�
=��p�A0(��E���8�\B�����p�Anff��p���{B�\                                    Bxe��v  �          A�z����HAAG��&{��B�{���HAr=q��ff���
B�.                                    Bxe��  �          A�����p�A:{�2ff�#�B����p�Ap(����H���HBۨ�                                    Bxe���  �          A��
���RAB�\�.�H�G�B�W
���RAv�R���R��=qB�Q�                                    Bxe��h  �          A�Q��>�RA
=�bff�\{B���>�RAa����(�B��H                                    Bxe��  �          A��H�)��A	���m���k�B�33�)��AY�'33�{B�                                      Bxe�ݴ  �          A�G��0��A�R�k33�fG�B�ff�0��A]p��#\)�z�Bʔ{                                    Bxe��Z  �          A��
��\A�H�mG��h�\B�����\A^=q�%p��(�B�                                    Bxe��   �          A�\)�p�@�G��{33�z�
BӮ�p�AR�\�9G��#�
B�\                                    Bxe�	�  �          A�{��
=@Å���HW
BҮ��
=AB{�Xz��?ffB�                                    Bxe�L  �          A�p�����@}p����
�3B������A%�n�R�\�HB�k�                                    Bxe�&�  �          A������@�  ����8RB�\)����AA���\���B�B��                                    Bxe�5�  �          A�{��33@�33��=q��B�(���33AO��Q���4(�B                                     Bxe�D>  T          A�(���G�@ڏ\�����B��H��G�AL(��UG��8(�B�u�                                    Bxe�R�  3          A�{��\)@��������B�B���\)AI�W��:�
B�aH                                    Bxe�a�            A�녿�(�@��H���B����(�AN�R�Qp��4(�B�aH                                    Bxe�p0  �          A�  �   @����
=��BҊ=�   AQG��N�H�1Q�B�\)                                    Bxe�~�  3          A�=q���@陚����B��f���AQp��PQ��2��B�u�                                    Bxe��|  u          A�=q��z�@�{����B��Ϳ�z�AO��QG��3��B�                                    Bxe��"  �          A�Q�ٙ�@أ���G��qB�.�ٙ�AJ�\�V�R�9�B�L�                                    Bxe���  T          A��
��ff@���������B�B���ffAN�R�Qp��4\)B��H                                    Bxe��n  �          A�  ���
@�Q������{Bр ���
AI�V�\�9�HB�\                                    Bxe��  T          A��׿�=q@�33��
=�qB��)��=qAE��\���@�B{                                    Bxe�ֺ  T          A��Ϳٙ�@�Q����R��B�G��ٙ�AG
=�[��>(�BÏ\                                    Bxe��`  T          A������@أ����\�fBҙ�����AJ�\�Y��;�Bģ�                                    Bxe��  T          A�  ��@�
=��33\)BҊ=��AJ=q�[��<(�Bą                                    Bxe��  
�          A��R��33@љ����H�)B����33AJ=q�c��@�RB���                                    Bxe�R  �          A����33@�����\)aHB��H��33A@z��l���J�Bã�                                    Bxe��  �          A��ÿ�
=@�z����R�B����
=A7��s33�R�
B�8R                                    Bxe�.�  �          A�  ��\)@�p����
L�B�Ǯ��\)A733�qp��Rp�B��                                    Bxe�=D  �          A������@���{�B�ff����A3��s��U��B�8R                                    Bxe�K�  �          A�(��n{@z�H���\� BθR�n{A&ff�~{�b�RB�=q                                    Bxe�Z�  
�          A�
=�Y��@�33��ff�B���Y��A0(��y��[  B��)                                    Bxe�i6  �          A�����  @�=q����p�Bҳ3��  A/��{
=�[\)B�                                    Bxe�w�  
�          A�(����@�\)���\)BΞ����A.�H�}G��]
=B�{                                    Bxe���  �          A�
=��  @Fff���R (�B��
��  A(�����m�HB=                                    Bxe��(  "          A�  ����@~�R��Q�\)B՞�����A(�����R�c�
B�                                      Bxe���  �          A�zῘQ�@�������p�BΣ׿�Q�A733�|���W��B��
                                    Bxe��t  "          A��ÿ�
=@�{��G��\B͏\��
=A9��|���V�B��{                                    Bxe��  "          A��\��(�@�
=���\\B�=q��(�A,(���z��a�B��                                    Bxe���  �          A����z�@�p�����L�Bծ��z�A2�R��G��\�\B�p�                                    Bxe��f  
�          A�Q����@
�H��ff¢��C�ÿ���Aff��{�x�RB̽q                                    Bxe��  �          A�Q��G�?�Q����R£�CE�G�A
�H����{z�B�B�                                    Bxe���  �          A�(��p�@z���=q¢�C�f�p�AQ���Q��y�B�W
                                    Bxe�
X  �          A������@$z���  �
C�f���A����H�r��B�\                                    Bxe��            A�Q����@�  ��B�R���A%�����bG�B�                                    Bxe�'�  C          A��\�G
=@������H��B��R�G
=A'
=�33�^��B�33                                    Bxe�6J  
�          A��\�XQ�@�����33��B�Q��XQ�A/\)�x���VG�B�B�                                    Bxe�D�  T          A�z���G�@�G����\B��)��G�A;��lz��H
=B�\                                    Bxe�S�  
�          A��\�mp�@�z�����k�B��f�mp�AO
=�c33�:(�B�                                      Bxe�b<  
�          A�\)���A*=q�^�R�Dz�B�
=���Amp��G���\)B�Ǯ                                    Bxe�p�            A��\���AYG���R��=qB�����A�{���H�MG�B�                                    Bxe��            A�(�����AD(��A��"  B�� ����A{\)���H���RB��                                    Bxe��.  T          A��H��ffA:�H�C
=�#{B�� ��ffAs
=���
��{B�G�                                    Bxe���  
�          A�p��  A@  �Ap����B����  Aw33����Q�B�.                                    Bxe��z  u          A���Q�A>�\�L(��)B�q��Q�Ayp���33��=qB���                                    Bxe��             A�z����A%��G��$�Ck����A`���{��33B��H                                    Bxe���  �          A�=q�\)A�R�b�\�@��C��\)AT  �#��G�B��                                    Bxe��l  T          A�G��&�\A8Q��8����C��&�\Al����������B�G�                                    Bxe��  �          A�{�&ffAQ��'�
� Q�C h��&ffA~�R��33�{�B�L�                                    Bxe���  
�          A���AqG��ٙ���B���A�  ��p��l��B�#�                                    Bxe�^            A������A|z���G����B�=���A�33��\)�\(�B�aH                                    Bxe�  �          A�
=�	G�Aw\)��������B�\�	G�A�ff�z�H�>�RB��                                    Bxe� �  T          A�ff���Ar�H�ָR���B�(����A�zῑ��_\)B��                                    Bxe�/P  T          A�{���An�\������RB�����A�  �����RB���                                    Bxe�=�  �          A�Q��Q�At����z�����B�\�Q�A��׿\��
=B�B�                                    Bxe�L�  �          A����
=A~=q��
=����B�\�
=A��
�s33�2�\B�#�                                    Bxe�[B  �          A�ff�(�AxQ���p���z�B��(�A��H�޸R���
B���                                    Bxe�i�  �          A�(��A�
=������B���A�{�����G�B���                                    Bxe�x�  
�          A��R�G�A�  ��(���  B�\�G�A��H���R�fffB���                                    Bxe��4  �          A��\�z�A�Q���=q����B�.�z�A�=q=�\)>B�\B�L�                                    Bxe���  "          A�\)� ��A�
=��33�vffB�Q�� ��A��?�?�p�B��H                                    Bxe���  	�          A��!�A����z��k�B�Q��!�A��?=p�@
=B��                                    Bxe��&  �          A�=q��A�(���ff��z�B�33��A�Q켣�
�uB�L�                                    Bxe���  "          A���!G�A��\��G���B���!G�A���
=��Q�B�L�                                    Bxe��r  �          A���"�RA��
������ffB�aH�"�RA�\)�녿�{B��)                                    Bxe��  �          A��#�A���Ӆ���RB����#�A����J=q�{B�{                                    Bxe���  	�          A�G��*�HA�(���\)�s33B�ff�*�HA���>�33?��\B�                                    Bxe��d  T          A����+
=A�G���ff�o\)B�aH�+
=A�G�@���AF�HB���                                    Bxe�
  �          A���� ��A����#�
���B�aH� ��A�(�@B�\A
=B��                                    Bxe��  
�          A��R�{A�ff�9���z�B�{A��\@1G�A=qB�\                                    Bxe�(V  u          A�  ��A��R�-p�� (�B�33��A�ff@<��A�B�Q�                                    Bxe�6�  T          A�{��
A��=q��B�u���
A��R@L(�AffB��)                                    Bxe�E�  �          A�\)��RA��5��Q�B�\)��RA��
@1�A=qB�W
                                    Bxe�TH  T          A�(��&{A�33�����{B��f�&{A�(�@J=qA33B�L�                                    Bxe�b�  
�          A���,  A��>k�?.{B�L��,  Aw�
@�=qA���B��                                    Bxe�q�  �          A��H��
A�p���p���\)B����
A�33@l(�A+�B��                                    Bxe��:  u          A��� z�A����
=���HB�u�� z�A��@��RAC
=B��
                                    Bxe���  
�          A�(��-��A��
����P��B���-��A��@�=qAH  B���                                    Bxe���  "          A�z��#�A�녾�녿�z�B�W
�#�A��@�p�An�HB�R                                    Bxe��,  "          A��
�(��A���Ǯ����B�q�(��A��@��\Ak�B�(�                                    Bxe���  "          A���%�A�Q��G����\B���%�A�=q@�G�Aj=qB���                                    Bxe��x  �          A�p��%G�A����8Q�
=qB�B��%G�Azff@��HAs
=B��H                                    Bxe��  �          A�ff�%p�A�(��\)��(�B��)�%p�Aw\)@���As33B��                                    Bxe���  �          A�Q���HA���>�p�?��B�#���HA}��@��RA��\B��                                    Bxe��j  T          A��H��
A�\)?��@K�B�G���
A
=@θRA�Q�B�                                    Bxe�  �          A�33�(�A��R@<��A{B�ff�(�Ah��@�\)A�33B�L�                                    Bxe��  
�          A�  � Q�A|Q�@���AZ{B��)� Q�AXz�A��A�
=B�
=                                    Bxe�!\  �          A�  ���A�  @1�A��B�(����Ah��@���A�33B��                                    Bxe�0  
�          A��H�$��A��R@��
AM�B�L��$��A^ffA\)A�(�B�#�                                    Bxe�>�  
�          A�{�(��A~{@�z�AC�
B�8R�(��A\z�A�RAمB��f                                    Bxe�MN  
�          A����(  Ax��@�p�A_�B�\�(  AT��A��A�Q�C G�                                    Bxe�[�  �          A���!A�ff@L��A��B�ff�!Ad  A ��A�{B��3                                    Bxe�j�  
�          A��� z�A��\@@  A(�B��� z�AeG�@��
A�B���                                    Bxe�y@  �          A�����A�Q�@!�@��HB�����Ak
=@�\)A���B�u�                                    Bxe���  �          A���#�A��\@
=@��B���#�Ah��@�\)A��B�33                                    Bxe���  �          A����A��@�@��B�Q���Aj=q@�\A��B��{                                    Bxe��2  �          A���
=A��@��@�z�B��
=An=q@�A�  B��                                    Bxe���  
�          A����{A�@@�  B���{ApQ�@�=qA��HB�W
                                    Bxe��~  
�          A���
=A�@\)@�{B�{�
=As�@��A�  B�=q                                    Bxe��$  
�          A�G��=qA��\?�z�@�\)B���=qAr�H@��A�G�B�                                     Bxe���  �          A�Q����A�33?�@�(�B�q���At��@�=qA�B�                                    Bxe��p  �          A��
��
A�ff@�R@�p�B�L���
As�
@�  A�B�{                                    Bxe��  "          A��
���A�=q@   @�RB�����As�@�A���B�u�                                    Bxe��  �          A�ff�A�(�@5A�B�
=�Aq@��A��B�33                                    Bxe�b  
�          A�����HA�(�?�Q�@dz�B�k���HAv{@���A��\B�                                      Bxe�)  
�          A���z�A���?��?�33B���z�Az�R@��A���B�{                                    Bxe�7�  
�          A�p��33A��\?�p�@��B���33AtQ�@��A��\B�#�                                    Bxe�FT  �          A�� (�A���@4z�AG�B��� (�AmG�@���A��B�                                    Bxe�T�  
�          A�G��=qA���@8��A��B�
=�=qAm�@�ffA�
=B�p�                                    Bxe�c�  �          A����
A�ff@33@�G�B����
Aup�@�A��HB�                                     Bxe�rF  
�          A��� ��A�  @a�A%��B��� ��Ak
=A��A�=qB��q                                    Bxe���  
Z          A��"�HA�=q@[�A (�B�8R�"�HAl  A\)A��HB�33                                    Bxe���  �          A����$z�A��@
=q@��HB��f�$z�Aq��@�\)A�G�B��                                    Bxe��8  �          A��
�$z�A�33@#33@�B�aH�$z�Ar=q@�z�A��B�\)                                    Bxe���  "          A�{�%p�A�G�@(�@�33B���%p�Ar�H@���A��HB��                                     Bxe���  "          A��
�'33A�z�@��@�(�B�=�'33Aqp�@�A�=qB�k�                                    Bxe��*  
Z          A��
�)p�A��\?�{@�B�(��)p�Au��@�A�=qB�.                                    Bxe���  
�          A�G��)p�A�p�@ff@��
B����)p�AqG�@��HA�
=B�(�                                    Bxe��v  
�          A�
=�'�A��@
=q@�=qB����'�Aq��@��A��RB��=                                    Bxe��  "          A���#
=A��\@'�@�B�33�#
=Aq�@��
A�Q�B�#�                                    Bxe��  �          A�p��$Q�A��R@&ff@�\B� �$Q�Aq��@��HA��B�k�                                    Bxe�h  �          A�  �!p�A��@&ff@���B��f�!p�ApQ�@陚A��
B�Ǯ                                    Bxe�"  
l          A�=q�{A���@*=qA Q�B�p��{Ak�@�Q�A�p�B�W
                                    Bxe�0�  t          A�33�ffA���@��@��
B���ffAn�H@�33A�(�B�                                    Bxe�?Z  T          A�
=��\A�?�{@U�B�Ǯ��\Aw
=@��HA�(�B�\                                    Bxe�N   �          A���"�HA�Q�?�R?�{B�  �"�HAv�\@��A�Q�B��H                                    Bxe�\�  �          A�G����A��R@Q�@�(�B�8R���Av�H@�(�A���B�k�                                    Bxe�kL  �          A�z��A��?�\@��\B�k��Av�R@ϮA�33B�(�                                    Bxe�y�  T          A�ff��
A��
��Q쾏\)B�k���
A�ff@�
=AjffB�W
                                    Bxe���  
Z          A����p�Ah���	p���G�B�u���p�A������L��B�8R                                    Bxe��>  
�          A�z���A[��1p���B����A�����ff��p�B�\)                                    Bxe���  �          A�  ��ffAg
=�����B�u���ffA����{��  Bܽq                                    Bxe���  �          A����G�A��H����m��B���G�A��\���
���
B�                                    Bxe��0  T          A���33A��!G���  B��33A�(�@(�@�
=Bힸ                                    Bxe���  "          A��\��HAw\)�A���HB�{��HA{
=?�(�@�=qB�W
                                    Bxe��|  
�          A�=q��Ax���6ff��B���A{\)?�z�@�z�B���                                    Bxe��"  
�          A�Q���Ax���O\)� ��B�B���A}�?��\@|(�B�W
                                    Bxe���  T          A�33�3�A\Q���ff�`��C0��3�Ag��Ǯ��p�B���                                    Bxe�n  T          A�p���RA��
����VffB��f��RA~=q@`��A-G�B���                                    Bxe�  �          A�=q�A�\)�
=���
B�\�A~�R@���AFffB�\                                    Bxe�)�  
�          A��H�
=A}녿@  ��
B����
=Aw33@mp�A8(�B�.                                    Bxe�8`  "          A���ffA|��>���?p��B�(��ffAqp�@�{AiG�B��=                                    Bxe�G  T          A�  �#33Az�H>k�?8Q�B�{�#33Ap  @��\Ab�HB�p�                                    Bxe�U�  �          A����%�Az�\?���@W
=B��%�Ak�
@���A��B���                                    Bxe�dR  �          A�ff��A��H?��R@n�RB�{��Ay�@���A�p�B��                                    Bxe�r�  T          A��\��A���@�R@�B�Q���Axz�@�G�A���B�8R                                    Bxe���  f          A�{�Q�A��@�@���B��)�Q�Au@�=qA���B잸                                    Bxe��D  B          A�33��
A�=q��R��B�B���
AyG�@s33A<Q�B�                                    Bxe���  
�          A�33�p�A|Q쿳33����B���p�Ax��@>�RA
=B���                                    Bxe���  �          A����=qA{
=�%� Q�B�#��=qA|��?��@�G�B�Ǯ                                    Bxe��6  "          A�33�z�A����p��ڏ\B���z�A���@@�
=B�.                                    Bxe���  �          A�33�(�A|�Ϳ����G�B�k��(�AyG�@=p�A=qB��                                    Bxe�ق  �          A���
=Ax  �g��3\)B��
=A~ff?8Q�@p�B�R                                    Bxe��(  
�          A����"�RAv=q������33B��H�"�RAr�H@8��A  B���                                    Bxe���  
Z          A�Q��8��Ag\)>���?k�C ���8��A]p�@�{AQ��C�{                                    Bxe�t  �          A����33AV�R�ȣ����B���33Ai���H�G�B�=                                    Bxe�  	�          A�(���Q�AB{�f�R�;=qB�3��Q�Ax���)��  B�8R                                    Bxe�"�  
Z          A��R��p�AEG��`z��4{B�Ǯ��p�Azff�#\)��B�(�                                    Bxe�1f  �          A�p���(�AJ�R�S��&G�B�q��(�A|  ����B�#�                                    Bxe�@  	`          A�z���
=APQ��PQ��!�
B����
=A�(�����
=B��H                                    Bxe�N�  �          A�
=��AS\)�XQ��(�B�aH��A��R�(���p�B�                                      Bxe�]X  "          A��R��\)AP���R�\�#�RB�8R��\)A��R�\)��z�B�L�                                    Bxe�k�  �          A�=q���
AH���j=q�8��B��)���
A��,��� �B�z�                                    Bxe�z�            A�(���33AS��[��*��B�3��33A����
���
B�W
                                    Bxe��J  �          A�����z�Al���4  �=qB����z�A��
��ff��  B���                                    Bxe���  �          A�
=�=qAo
=�%p�����B���=qA�
=�������RB�.                                    Bxe���  �          A�����
Amp��  ��=qB�����
A�
=������p�B�R                                    Bxe��<  T          A��\���A}�������B�W
���A���^{�B�                                    Bxe���  �          A�=q���A�����z�����B�aH���A�(��B�\��RB�Q�                                    Bxe�҈  �          A�(��{A|���
=��  B�
=�{A�33��  �S�B�z�                                    Bxe��.  �          A�Q����Ay�����=qB�B����A����p��\��B�                                    Bxe���  4          A�p��*�HA�33�������RB�\�*�HA�\)��ff��  B�{                                    Bxe��z  �          A����*�RA�33�����\)B�\�*�RA��ÿ�z��w�B�3                                    Bxe�   �          A��H�,��A�  �׮���HB�u��,��A�p��  ��{B��H                                    Bxe��  
�          A���7�
A��R��33�mB�k��7�
A�G��fff�{B��)                                    Bxe�*l  �          A��
�4��A�����Q��hQ�B���4��A�33�@  ��\B�L�                                    Bxe�9  "          A�  �0��A��\��  ����B����0��A�ff���R���B��                                    Bxe�G�  T          A����)G�A���\)���\B����)G�A�{�-p�����B��)                                    Bxe�V^  �          A�
=�2�HA�G���G��t��B���2�HA�{��ff�6ffB�\)                                    Bxe�e  
�          A���+33A�z���\���B���+33A����n�R�"�RB��)                                    Bxe�s�  
�          A�G��.�HA�  ��R���B�\�.�HA����AG��\)B���                                    Bxe��P            A�z�� z�A��\�=q��G�B�ff� z�A�=q�����>�RB�                                    Bxe���            A�\)� (�A�������G�B�q� (�A�����\�:�RB���                                    Bxe���  
�          A�33�'\)A��  ����B�\�'\)A�{�qG��"ffB                                    Bxe��B  �          A��"�HA�z�����ǅB�#��"�HA�ff���\�Ep�B�#�                                    Bxe���  �          A�(���RA�Q��$(���p�B�ff��RA��R���H�{�B뙚                                    Bxe�ˎ  
�          A�=q��A{
=�2�\��B���A�����
���HB�G�                                    Bxe��4  
�          A��R�z�A}p��2=q����B�{�z�A����ڏ\����B�z�                                    Bxe���  "          A�����A{�
�5p����B����A��\������RB�\)                                    Bxe���  	�          A�=q��HAw��6�R� (�B�\��HA��\��R����B�                                      Bxe�&  
Z          A��R�=qAw
=�6{���\B�G��=qA�=q��{���B�(�                                    Bxe��  
�          A�����
Av{�;
=���B����
A�Q���Q����\B�aH                                    Bxe�#r  �          A�33��ffA�{�7
=���\B�W
��ffA�z���ff��\)B���                                    Bxe�2  "          A��H��z�A�33�2�H��(�B���z�A�
=�Ӆ����B�
=                                    Bxe�@�  �          A�
=�(�A���733���\B虚�(�A�{�߮���B���                                    Bxe�Od  �          A������A���<z��\)B����A��R��ff��G�B��                                    Bxe�^
  
�          A����
=qA�
=�?
=�z�B�aH�
=qA�z���(���p�B��                                    Bxe�l�  "          A��R�{A{\)�E�	��B����{A�  �ff����B��f                                    Bxe�{V  
�          A������Ay���@���Q�B�����A��\�������
B�                                    Bxe���  
�          A�����Avff�?�
�  B�L���A�����z����B��)                                    Bxe���  "          A����ffAz{�:�R� ��B��R�ffA��������
=B���                                    Bxe��H  "          A�
=�!p�A|(��5G����B�B��!p�A�=q����G�B�u�                                    Bxe���  
(          A�33��A33�8�����\B����A�(���33����B�L�                                    Bxe�Ĕ  
�          A�\)��
A���>�H���B�R��
A��R�����B�aH                                    Bxe��:  �          A�\)�G�A��H�6�H��p�B�.�G�A��H��G����B���                                    Bxe���  
Z          A�(����A��R�5����B�p����A����ᙚ��33B��
                                    Bxe���  �          A�
=��HA�{�<z��\)B�3��HA�������(�B��f                                    Bxe��,  �          A�33��A���@(��  B�(���A�
=��G�����B��                                    Bxe��  
�          A�33�(�A}�E���
B����(�A��R�
=���B�{                                    Bxe�x  "          A�G��
{A}p��F�R�
33B�.�
{A��R�����=qB�k�                                    Bxe�+  "          A����\A���E��	�B�p���\A�p���H���HB�.                                    Bxe�9�  T          A���� ��A����F�\�
ffB�8R� ��A�z��  ���B��                                    Bxe�Hj  T          A�(����Aq��Y�����B�8R���A�G��=q��ffB�                                    Bxe�W  �          A������Ax(��Tz��{B�\���A����(�����B��                                    Bxe�e�  4          A��\��RAxQ��X���z�B��)��RA�(��z���
=B��                                    Bxe�t\  	�          A�����Az{�R=q�(�B�Q���A�=q��ŅB�{                                    Bxe��  
�          A�=q��Ayp��S��z�B����A�  �\)��(�B��                                    Bxe���  
(          A��R�ᙚAv�\�\z��p�B�(��ᙚA��������\)B�W
                                    Bxe��N  �          A����
=Aw33�]p����B䙚��
=A�����Q�B��
                                    Bxe���  
�          A��R��z�Arff�e�$�\B�R��z�A��\�'33����B��                                    Bxe���  �          A��R��G�Ay�]����B�u���G�A���G����
B�(�                                    Bxe��@  �          A��\��ffAy�Z�H�ffB�u���ffA����33��G�B��                                    Bxe���  "          A�33���An�H�h���&B�����A�
=�+����B��)                                    Bxe��  �          A�����HAr�R�hz��%�
B�aH���HA��H�*=q��{Bڙ�                                    Bxe��2  �          A�\)��Q�An=q�n�\�+z�B����Q�A�\)�1p���33B�\                                    Bxe��  �          A�Q���G�Ad���u�3p�B�
=��G�A����:�R���Bس3                                    Bxe�~  �          A�ff����Al���o\)�-(�Bޔ{����A��\�2�H���\B��)                                    Bxe�$$  
�          A����At���a��"(�B�  ��A����$  ��G�Bٮ                                    Bxe�2�  �          A�  ���RAw33�g\)�&��Bה{���RA��R�(����\B�B�                                    Bxe�Ap  �          A����Q�Az{�Xz���B�8R��Q�A�Q������HB�(�                                    Bxe�P  
�          A�\)���Ayp��^ff���B�p����A����   ��  B�Ǯ                                    Bxe�^�  �          A�����=qAX(���=q�B  B�z���=qA�Q��H����B��H                                    Bxe�mb  
�          A��׿�
=A	�����p�B��H��
=AJff����Q(�B��                                    Bxe�|  �          A��׿�
=A
=���R�w=qBǳ3��
=AY��{
=�C�B��                                    Bxe���  �          A��\��ffA=q���\=qB�#׿�ffAC���  �V�B�#�                                    Bxe��T  T          A�ff����@�=q�����HBʮ����A>�H����Z�\B���                                    Bxe���  "          A��;��@�����\�qB�uþ��A5���cB��3                                    Bxe���  "          A����aG�@��R��ffB�\�aG�A<Q����R�W�\B�L�                                    Bxe��F  �          A����p�@ə�����B�����p�A'������e=qB�\)                                    Bxe���  
�          A��׿�(�@��R��p��Bϊ=��(�A{���\�}=qB�                                    Bxe��  "          A���\)@\)��p�¦B�B���\)@��������B�Q�                                    Bxe��8  �          A�녿333@H����p�¡�HB��333@�G�����=B��{                                    Bxe���  "          A�(��E�@��\���H��B���E�A Q������r\)B���                                    Bxe��  T          A��ÿc�
@�ff���{B�{�c�
A0  ��(��kp�B�8R                                    Bxe�*  4          A����\)A(���  �B���\)AL�������R��B�8R                                    Bxe�+�  
�          A���A
=�����u�\Bˣ���A\�����C�\B�(�                                    Bxe�:v  
�          A����H��AH(����W��B�(��H��A��R�h���&Q�B�\                                    Bxe�I  5          A�p����HAs�
�~�R�5  B�{���HA����C33��HB�(�                                    Bxe�W�  s          A�����(�A�Q��n=q�&�B�.��(�A�G��0z���Q�Bπ                                     Bxe�fh  T          A�\)����AqG��z{�3��B��)����A����?\)��HBθR                                    Bxe�u  
�          A�z����
Av�R�s\)�.p�B�����
A��H�7�
��G�B�L�                                    Bxe���  
�          A�z���=qA�33�g33�#��B�Q���=qA�33�)�����B�Q�                                    Bxe��Z  �          A�(����Ax���k�
�(
=B����A���0Q���(�B�Ǯ                                    Bxe��   
(          A�\)����Aw��q��,(�B�B�����A�
=�6�R��(�B��                                    Bxe���  �          A��H���Apz��y���1�B�u����A�ff�?�
�(�B�ff                                    Bxe��L  g          A�p����
A|Q��p  �(  Bٙ����
A�
=�4(���
=B�(�                                    Bxe���  �          A�����z�Ag���ff�7Bޙ���z�A����H�����Bה{                                    Bxe�ۘ  �          A�ff��Q�At���w��/��B֏\��Q�A�{�=G�� Q�B�8R                                    Bxe��>  	�          A�����p�A���k
=�$(�B��f��p�A�G��.=q���B�(�                                    Bxe���  
�          A�����
=A��P�����Bӽq��
=A����Q���=qB��                                    Bxe��  T          A�(���{A�z��>ff�ffB�\��{A�p��������B��
                                    Bxe�0  �          A����\)A���$����=qB�����\)A����p��tz�B��H                                    Bxe�$�  T          A�z���{AUG��x  �<p�B�k���{A��\�E��ffB��H                                    Bxe�3|  "          A�z�����A{��  �mz�B������ATz��|���@�B܅                                    Bxe�B"  �          A�(��   AUG������N�B���   A���^{�z�B��q                                    Bxe�P�  
�          A��R���AFff��  �X�RB������A}��g�
�(��B�.                                    Bxe�_n  s          A������AK�
��(��Y�HB�=q����A�  �n�H�)��B�                                    Bxe�n  �          A�
=�$z�A<����=q�c��B̙��$z�Av�H�z{�3�B��                                    Bxe�|�  5          A�  ��AN�\����W�HBƮ��A�33�n{�'�HB�                                    Bxe��`  A          A�Q��,��AJ{��33�ZG�B���,��A�33�q���*�RB���                                    Bxe��  
�          A������AQ���z��U  B�B����A�{�j�H�%\)B��f                                    Bxe���  �          A��
�AG�A_������CffB�\)�AG�A��R�Qp��{B�                                    Bxe��R  T          A�33���A�=q�L���\)B�=q���A������ƸRB�\                                    Bxe���  �          A��H��  A���_33��B�  ��  A�{�$Q���Q�B��                                    Bxe�Ԟ  "          A������RA����a�� �\B����RA�33�&�H��=qB��                                    Bxe��D  
Z          A��H���HAy�j�R�(��B�����HA�z��1��z�Bγ3                                    Bxe���  
(          A�=q�tz�Ar�\�r=q�0G�B�L��tz�A��:�R��RB�                                    Bxe� �  "          A�(��g
=An�R�v�H�4�\B�=q�g
=A�Q��@Q��  B���                                    Bxe�6  �          A���`  AxQ��k33�*�HB�ff�`  A��2�H���\BɸR                                    Bxe��  �          A���\(�A�\)�_33� ��B��\(�A����%���  B��
                                    Bxe�,�  
�          A���qG�A�p��Nff��B���qG�A��=q��(�B�33                                    Bxe�;(  
�          A���}p�A���P(��z�B�\)�}p�A���Q���z�B�G�                                    Bxe�I�            A��\�n{A���Zff��B��H�n{A�G�� ���ޏ\Bʔ{                                    Bxe�Xt  
�          A�=q�p��A�=q�<(���B̅�p��A��\��\)����B�                                      Bxe�g  T          A������\A�G��)���RB�\���\A�����  ��33B�z�                                    Bxe�u�  "          A������\A��2�\���\B��H���\A������
����B�8R                                    Bxe��f  
�          A�G���A�{�C��ffB�k���A�
=��\���B�u�                                    Bxe��  
�          A�
=��ffA��R�=��z�B��
��ffA����   ��  B�{                                    Bxe���  
�          A�  ��p�A��\�9����
B�aH��p�A�z���=q���\B�=q                                    Bxe��X  
�          A����p�A���5p�����Bܽq��p�A�
=��
=����B�{                                    Bxe���  
�          A��
��G�A�33�2�R��\)B�=q��G�A�Q�������RB��H                                    Bxe�ͤ  T          A��ǮA�  �)�����B�G��ǮA�{������G�B�=q                                    Bxe��J  
�          A�\)����A����+
=��RB�������A�����\)��\)B�                                      Bxe���  T          A�
=��\)A��\�-p���\B�z���\)A�����(�����B���                                    Bxe���  
�          A��\��p�A�33�)���噚B�����p�A�33��z���  B�p�                                    Bxe�<  "          A�=q��(�A�33�����33B�.��(�A��
�����tz�Bҽq                                    Bxe��  "          A�\)���A�(��p���Bԙ����A�{���H�aB�W
                                    Bxe�%�  
�          A����p�A��(�����BΏ\��p�A������H���B�aH                                    Bxe�4.  �          A������HA�p��33��Q�B�\���HA�Q���������B��
                                    Bxe�B�  �          A���Q�A��\���Q�B�u���Q�A�  ��(��ZffB�z�                                    Bxe�Qz  
�          A�{���HA�33�{��z�B�\)���HA�����z��Z�RB�u�                                    Bxe�`   �          A�ff���A�=q�
=���B�����A�p���\)�S�B���                                    Bxe�n�  
�          A�Q�����A����Q��îBӔ{����A������V�HBр                                     Bxe�}l  �          A�Q����A�������Bӊ=���A�\)�����aG�B�ff                                    Bxe��  �          A����\A����\)��{B�=q���\A��\��  �h  B�{                                    Bxe���  �          A����Q�A��
��R���B�W
��Q�A�����hQ�B��                                    Bxe��^  T          A������RA�� ����(�B�Ǯ���RA����ȣ���
=B�{                                    Bxe��  �          A�=q��\)A�  �=q�ڸRB��f��\)A��R��33��B�=q                                    Bxe�ƪ  "          A�����\)A��
�
=��ffB��H��\)A��
���w
=B�aH                                    Bxe��P  T          A�=q��33A����p����\B�Q���33A��������Dz�B�#�                                    Bxe���  �          A�Q��љ�A�{�
=���B����љ�A�  �����?
=Bخ                                    Bxe��  "          A�z���Q�A��������B�L���Q�A�  �tz��%G�B��                                    Bxe�B  �          A�=q��\A�33�����B�=q��\A�33�I���(�B�33                                    Bxe��  
�          A�Q���\)A�������B���\)A���Y���33B��f                                    Bxe��  �          A��\��ffA��
������
B�k���ffA��R�(����HB��H                                    Bxe�-4  
�          A�����G�A����{���B�W
��G�A��
�G
=�{Bي=                                    Bxe�;�  
�          A�
=��=qA�����\)���Bܣ���=qA�  �(�����HB���                                    Bxe�J�  �          A�����A���  ���Bۨ���A����(����\B�                                    Bxe�Y&  �          A�ff���HA�����33�HQ�B�ff���HA��
�J=q���B�Q�                                    Bxe�g�  
Z          A�\)��Q�A�z���
=�e�B�(���Q�A�������hQ�B�#�                                    Bxe�vr  �          A��H�G�A�G��$z���B���G�A�z�?�=q@AG�B���                                    Bxe��  �          A������A�\)�33�θRB��H���A�=q?�G�@a�B��                                    Bxe���  T          A����&�HA���?!G�?���B�  �&�HA�{@uA3�B�\)                                    Bxe��d  �          A��\�
=A��H�u�8��B��H�
=A��@��@ڏ\B�B�                                    Bxe��
  �          A�=q��33A�33�g
=�9��B�8R��33A�=q���ÿ���B�p�                                    Bxe���  �          A���ָRA�Q�У�����B�Ǯ�ָRA�z�?��
@�ffB�                                    Bxe��V  T          A����Ao�@~�RAPQ�B�\��A^�H@���A��\B��                                    Bxe���  "          A�Q���Ak
=@^�RA4��B�{��A\  @�\)A�=qB��\                                    Bxe��  
�          A�z����AlQ�@���A]��B����AZ�H@��A�\)B�                                      Bxe��H  "          A�����An=q?�=q@�  B�q��Ad(�@���Au�B��                                    Bxe��  T          A�z���HAk��J=q�&ffB�=q��HAi��@�\@�B��                                    Bxe��  �          A�Q���RAdz�@�@�{B��{��RAY�@��A~�RB�.                                    Bxe�&:  T          A�p��{Ab�\@��Ap�B��
�{AV�H@��A�Q�B��3                                    Bxe�4�  �          A�(��33AhQ�?k�@C�
B��R�33AaG�@j�HAD��B�W
                                    Bxe�C�  �          A����\)Ag�
@G�@�
=B��f�\)A]p�@��A~�HB�L�                                    Bxe�R,  �          A��H�
ffAV=q�>�R�)G�B��3�
ffA[\)�\��=qB�u�                                    Bxe�`�  �          A�����Ai��(����B�(����AiG�?�=q@��
B�G�                                    Bxe�ox  �          A�G��ffAv{��(�����B��ffAr�H@   A (�B�.                                    Bxe�~  "          A��
�33Aw
=>�>���B�\�33Ar=q@C33AQ�B�{                                    Bxe���  �          A�Q����Ax��?��@�B�p����Ap  @��A]G�B�8R                                    Bxe��j  �          A�Q���RAv�H@*�HAffB�����RAj=q@�G�A��B��                                     Bxe��  �          A��
���Am�@p  AG33B�u����A]p�@�\)A��B��
                                    Bxe���  
�          A����\)Aa������	��B�G���\)Ad��>��R?���B��                                    Bxe��\  
�          A�ff���\AQG������\B������\A[��Q���p�B�aH                                    Bxe��  T          A���z�AP(�?�\)@��\B��f�z�AH(�@s33AW\)C                                     Bxe��  T          A��H��\AQG�@z=qAZ{B�k���\AA@ə�A���B���                                    Bxe��N  T          A�  �
=AE��@�Q�A��\B�G��
=A.�\A�B 
=C �\                                    Bxe��  �          A��R�	G�AE�@��
A�\)B�u��	G�A/33A	G�A�G�C \                                    Bxe��  �          A���
�HAE��@���A��B�8R�
�HA.�\A�
B G�C ��                                    Bxe�@  "          A�z���
A<Q�@�33A�(�B�����
A$Q�A
=BffC�
                                    Bxe�-�  T          A��R��A?
=@��
A�\)B�aH��A'
=A�
BC�                                    Bxe�<�  �          A����\)A@��@�\)A��B�L��\)A)�ABp�CJ=                                    Bxe�K2  T          A�G����A/�@ۅA�  C����A  A��B��C(�                                    Bxe�Y�  �          A�(����A-G�@�G�A���C&f���A�RA33A���C&f                                    Bxe�h~  �          Ay��
=qA4��@���A�ffB����
=qA�HAz�B ��C�                                    Bxe�w$  �          A{
=�Q�A0(�@�ffA�ffC{�Q�A�\A=qA��\C�\                                    Bxe���  �          A�G��ffA,  @�=qA�G�C+��ffAffA
=B�CǮ                                    Bxe��p  �          A�z���A(��@�(�A�z�C����Az�A�B��C	ff                                    Bxe��  T          A|z��#\)A ��@�A�G�Cp��#\)A
�HA
=A�p�C�H                                    Bxe���  �          A��R�)G�A3�@�33A�{CG��)G�A33@��HA���C                                    Bxe��b  
�          A����
AdQ콣�
��\)B� ��
A`z�@%�A�RB�W
                                    Bxe��  �          A�����
AQp�@��Aa��B��H��
AA@�A�\)C 
                                    Bxe�ݮ  �          A���p�AHQ�@���A�\)C +��p�A3�A   A�\)C:�                                    Bxe��T  �          A������AN�\@�Q�A^{B��3���A?
=@��HA�(�C��                                    Bxe���  "          A����HAQ�@h��AI��B�����HAC�@���A��HC c�                                    Bxe�	�  �          A���$Q�AN�R@��A
=C }q�$Q�AC�
@��A��
C�                                    Bxe�F  �          A����9G�A/\)@�
=A���C�\�9G�A��@��A�{C�q                                    Bxe�&�  "          A�p��9p�A$��@�(�A�G�C
W
�9p�AA
�RA��
C�)                                    Bxe�5�  �          A����>ffA"�\@ÅA��RC���>ffA��@�z�A�  C^�                                    Bxe�D8  
�          A�\)�:�\@�
=A�
B  C��:�\@�{A"{BC                                    Bxe�R�  �          A�=q�6ff@�ffA+
=B�
C�\�6ff@�33A<  B.��C�                                    Bxe�a�  �          A����=�@陚A  B	p�CO\�=�@�p�A*�HB��CaH                                    Bxe�p*  
�          A�(��D��@�\)Ap�A�Q�C�\�D��@�
=A z�B�RC�                                    Bxe�~�  
�          A�33�A�@�{A��B  C�A�@���A,��B�\C��                                    Bxe��v  
Z          A�\)�M�@��A�A�RC�=�M�@��
A�RBp�Cc�                                    Bxe��  �          A����K33@��A	G�A��C�
�K33@�ffA(�B�CǮ                                    Bxe���  "          A�G��>�HA�A33A��Cn�>�H@�\)A!�BG�C��                                    Bxe��h  T          A���%p�A
�HA�RB=qC���%p�@ָRA5��B*��C�                                    Bxe��  �          A�33�=qA��A2�RB&�Cn�=q@�\AL  BE=qC	�q                                    Bxe�ִ  "          A�ff��p�@�(�AO�
BG
=C=q��p�@�33Ac�Bb�RC�                                    Bxe��Z  �          A����
=@�  AN�HBHp�CW
�
=@�  A_33B_�\C@                                     Bxe��   
�          A��R��@У�AI�B>�CxR��@��HAY��BS33C�                                    Bxe��  	�          A��
�#�
@�(�A4��B'p�C8R�#�
@�p�AG�B=��C5�                                    Bxe�L  T          A�=q�3
=A Q�AQ�B
�C^��3
=@�(�A-G�B!\)CJ=                                    Bxe��  
(          A��\�B�RA��@���Aأ�C�R�B�R@أ�A�HB�C�                                    Bxe�.�  
�          A����9�A=q@�G�A�=qC�\�9�@��A��BffC��                                    Bxe�=>  A          A��K�@�(�@�z�AۮC޸�K�@�33A�Bp�CL�                                    Bxe�K�  �          A���L��@�
=A(�B=qC
�L��@��A#�
B�C!                                      Bxe�Z�  "          A����N�R@���@�ffAݙ�C�\�N�R@ə�AQ�B=qC                                      Bxe�i0  �          A�Q��N�\@��@��
A�(�C��N�\@ȣ�A�HBp�C{                                    Bxe�w�  
�          A��H�@  A"�\@��HA�=qC�q�@  A
=Ap�A���C�                                    Bxe��|  �          A�G��O\)A=qA�HA��Cٚ�O\)@�  A��Bp�C\)                                    Bxe��"  �          A�\)�K�@���A{Bp�C:��K�@�A(��BffC��                                    Bxe���  �          A��R�I�A
{A\)A�Q�C�=�I�@�
=AffB�C��                                    Bxe��n  �          A�Q��H(�A33@��
A�p�C��H(�@�\)A�
A��Cs3                                    Bxe��  �          A��R�@(�AD��?z�H@QG�CL��@(�A>�R@N�RA,��C:�                                    Bxe�Ϻ  T          A�Q��Ap�AC
=?�(�@�ffCǮ�Ap�A:�\@z�HARffC
=                                    Bxe��`  
Z          A�z��F�HA@��@8Q�A33C�f�F�HA4��@�G�A���C	�3                                    Bxe��  �          A��R�@z�AG�
@7
=AC��@z�A<(�@��A�
=C��                                    Bxe���  �          A��
�@Q�AE��@8��A��C5��@Q�A9@��A�(�C�q                                    Bxe�
R  
Z          A���D��AAG�@,��AffC���D��A5�@��
A��C	B�                                    Bxe��  �          A��;\)AI��@:=qA��C���;\)A=��@�A�  C��                                    Bxe�'�  T          A����B=qABff@C33A!G�C�R�B=qA6{@�\)A��C�
                                    Bxe�6D  "          A�
=�8  AHz�@S�
A0Q�C���8  A;\)@��A��C}q                                    Bxe�D�  T          A�G��;
=AJ�\@�HA   C�R�;
=A@  @��RA{
=C@                                     Bxe�S�  
�          A�  �AA@Q�@
=@�z�C8R�AA6{@���At(�CǮ                                    Bxe�b6  �          A�(��@  AB{@(��Az�C���@  A7
=@�=qA��C^�                                    Bxe�p�  �          A��
�6�RAK33@�\@�(�C���6�RAA�@��\Av�RCn                                    Bxe��  
�          A�\)�,z�AQ@)��A�Cn�,z�AFff@�Q�A�C�                                    Bxe��(  T          A���&�HARff@�A ��C n�&�HAG�
@�  A�(�C�H                                    Bxe���  �          A�=q�*�HAP��@�\@��CJ=�*�HAF�\@���A}C��                                    Bxe��t  �          A���%�AV=q?�p�@�  B�8R�%�AN=q@z�HAU��C ��                                    Bxe��  
�          A�{��HAZ�H?�\)@s33B�����HAS�
@g�AEp�B�                                    Bxe���  
�          A����4  AG�
@C33A$��C��4  A;�@���A�z�C�
                                    Bxe��f  �          A�\)�9��AB=q@dz�A@  C��9��A4z�@��A�Q�C�=                                    Bxe��  "          A��\�6�HA@��@vffAP��Cu��6�HA2ff@�Q�A���C��                                    Bxe���  �          A�=q�3\)AB�R@~�RAXz�C���3\)A3�@��A���C��                                    Bxe�X  "          A���8z�A8z�@�{Af�RC��8z�A(��@ǮA�(�C	�                                    Bxe��  T          A�\)��=qA\)���
��=qB�=q��=qA|  @'�Az�B�                                    Bxe� �  
�          A��R��ffAxz��G���G�B����ffAup�@�HA��Bᙚ                                    Bxe�/J  "          A33�ҏ\AhQ�?E�@0  B��ҏ\Ab{@[�AF�\B��                                    Bxe�=�  
�          A|Q����
Ab�\?(��@�HB�q���
A\��@P��A?\)B��                                    Bxe�L�  "          Az=q��RA\z�?Ǯ@�  B�8R��RAT(�@��\Ar{B�\                                    Bxe�[<  T          A}����
Ac�?B�\@0��B�{���
A]G�@W�AE�B���                                    Bxe�i�  "          A|z���p�Ag
==#�
>�B�����p�Ab�H@,��Ap�B�q                                    Bxe�x�  "          Ax�����Af�R��\)����B��)���Af{?Ǯ@�Q�B���                                    Bxe��.  T          A|  ���Al  ���H���
B������Aip�@  A33B�8R                                    Bxe���  
�          AxQ����
Ajff�\)��Bڅ���
Ah  @	��@�
=B��H                                    Bxe��z  �          Aw��˅A_�����\B���˅A`��?z�H@hQ�B�                                    Bxe��   
�          Ay���
=Ak�����
=B�G���
=Ai�@p�A�HB٣�                                    Bxe���  @          Av{��Ac���
���RB�Ǯ��Ae?(�@��B�k�                                    Bxe��l  T          Av{���RA_�
�����B�����RAb�R>�33?�ffBᙚ                                    Bxe��  
�          Aw\)����A[\)�8Q��-G�B�B�����A`(���zῌ��B�W
                                    Bxe���  
�          As33��{AB=q��Q���=qB�ff��{AL(����Q�B���                                    Bxe��^  T          AuG���G�A?\)��33��33B�.��G�AK33�2�\�)��B�{                                    Bxe�  �          As
=��  AK��
=� ��B���  AN=q>�z�?�{B�                                      Bxe��  �          A������AMp���p����RB��q���AXz�����Q�B�
=                                    Bxe�(P  �          A�33�׮Arff���H��Q�B����׮Ao�@�Ap�B�u�                                    Bxe�6�  
�          A��H��Q�AuG���녿��B�#���Q�Ar=q@��A��B�                                    Bxe�E�  �          A�p�����Ag33���R����B�
=����As\)�+����B�Ǯ                                    Bxe�TB  "          A�(�����A`  �����Ə\B�3����Ap�������n{B�Ǯ                                    Bxe�b�  "          A������Ab�R��Q����HBފ=����AqG��`���E�B�(�                                    Bxe�q�  
�          A��H��33Ai���=q���HB�p���33Aw33�@  �&�HB؅                                    Bxe��4  �          A�\)���RAb=q�����ۅBڣ����RAup���  ��  B���                                    Bxe���  �          A�  �c33AT��� ���p�B��H�c33Ao33��z���p�BθR                                    Bxe���  �          A����Mp�ATQ��+\)�z�B�33�Mp�Ap��� ������B��                                    Bxe��&  "          A��R�|��AW�
�"�H��BԮ�|��Arff��\)��=qB�B�                                    Bxe���  �          A�ff���RAP���%p��B۸R���RAl  ���R���
B�p�                                    Bxe��r  
�          A��H��(�ATQ��#\)���B�p���(�Ao33��G����
B�ff                                    Bxe��  
�          A��R���AQ��(����B�u����Al����p���G�B�k�                                    Bxe��  �          A�33��(�AW33� Q����B�����(�Aqp���=q��
=B��                                    Bxe��d  �          A�
=����A]p������B�����Aup������G�B�p�                                    Bxe�
  
�          A�33��\)AZff�\)���B�33��\)As���
=���HB�z�                                    Bxe��  �          A�����\)Ab�R�����B܏\��\)Ay���������B�{                                    Bxe�!V  T          A��
��ffAn�H��{�ͅB�����ffA��H����x(�B�                                      Bxe�/�  
�          A�p���{Ai��=q����B�Ǯ��{Az�H��ff�\  B�R                                    Bxe�>�  

          A�33��Am���
=�x��B�
=��Aw33����Q�B�\                                    Bxe�MH  
          A��H��33Ap������up�B�\��33Az�R��p�����B�8R                                    Bxe�[�  
�          A�\)��p�Aip���Q�����B�3��p�Azff���
�X��B��                                    Bxe�j�  �          A�(���=qAd���
=q��RB�
=��=qAz�R�������B�aH                                    Bxe�y:  T          A��
���
AYG��,  �ffBُ\���
Au����ӮB�p�                                    Bxe���  T          A�����(�AY���2�\��\B�Ǯ��(�Aw��{���B��f                                    Bxe���  T          A�z���A]���,z���RB՞���Azff���R��\)B��                                    Bxe��,  �          A�{����A`Q��#��
z�B�������A{\)��z����B�                                    Bxe���  "          A������Aep����G�B�������A~�\��
=��p�B�=q                                    Bxe��x  T          A�  ����Adz��Q����B�u�����A}G���z����Bٳ3                                    Bxe��  
�          A�����AXz��,  ��
B�G����AuG���
=��  B�(�                                    Bxe���  
�          A�(����HAR�R�#33��RB�p����HAm���  ���B�Q�                                    Bxe��j  �          A�����\AXz����33B�
=���\Ar=q��
=���B���                                    Bxe��  T          A�G���
=A_�
�G��\)B�#���
=Ax���׮��G�B�p�                                    Bxe��  
Z          A�{���Ab�H��\�   B�L����A{\)�У���z�B׸R                                    Bxe�\  
�          A����z�Ad  �����
=B➸��z�Az{����p�B��                                    Bxe�)  
�          A�33���
A\���  ��(�B�=q���
As���ff��B�                                      Bxe�7�  
�          A����Q�AZ�R�G����B虚��Q�Aq�������\B�.                                    Bxe�FN  �          A�=q���AW��
=���B鞸���Ao
=��{��G�B�                                      Bxe�T�  �          A���{AVff�z���B�����{AmG���G����B�(�                                    Bxe�c�  "          A�  ��{AW��G���G�B�����{AmG����\����B�L�                                    Bxe�r@  �          A������A^{���
��B��
���Ao�
����d  B��                                    Bxe���  T          A�
=@ ��@y�����L�Be�@ ��@�ff��G�#�B���                                    Bxe���  6          A�z�@:=q@K���=qB>�@:=q@љ������HB��                                    Bxe��2  �          A�=q��Q�@�ff��\)  B�aH��Q�A#
=�g��[p�B���                                    Bxe���  T          A����L��@�����(��y  B����L��A,  �c\)�N�
B�#�                                    Bxe��~  
�          A�33�ffA{�}G��o�Bѳ3�ffA<Q��\���C��Bʙ�                                    Bxe��$  r          A���Z�HA,���e��N��B�(��Z�HAU��?33�#�\Bг3                                    Bxe���  "          A�Q�#�
A	��
=�w��B�z�#�
A9G��bff�J��B�W
                                    Bxe��p  �          A�G�@3�
@�{��z��)Bl\)@3�
A���z��z�
B���                                    Bxe��  "          A��R@<(�@�(�����Bf=q@<(�@����  �z��B���                                    Bxe��  �          A�G�@0��@�ff���\�=Bn{@0��A z���z��y��B�                                    Bxe�b            A�z�@>{@�{��33��Bf��@>{A   ��33�x��B�B�                                    Bxe�"  �          A��@/\)@�ff���H.B��@/\)A!��r�H�]z�B���                                    Bxe�0�  T          A��H@-p�@�\)�����|G�B�u�@-p�A1G��k��P�HB�z�                                    Bxe�?T  "          A���@�H@�  ��ǮB��=@�HA   �x���b(�B�                                    Bxe�M�  "          A�(�@<(�@���ffB�B�z�@<(�A
=�zff�a�HB�\                                    Bxe�\�  "          A�G�@p�@�p���p���B�z�@p�Az������jffB�                                      Bxe�kF  "          A�\)@*�H@�\)��{
=B��@*�HA�\���
�mG�B�\)                                    Bxe�y�  �          A�=q@>{@�G���
=z�B{p�@>{A(���
=�nB�ff                                    Bxe���  �          A�
=@>�R@�(���8RB|ff@>�RA�����n�B���                                    Bxe��8  T          A�{@\��@������W
Bp�H@\��A(�����k�B�(�                                    Bxe���  T          A�p�@7
=@�Q����{B��@7
=A�����R�k33B�p�                                    Bxe���  "          A��\@!�@��H��  =qB�  @!�A"{���\�h�RB���                                    Bxe��*  T          A���@��@����H�=B�@��A)����R�b�B��\                                    Bxe���  �          A�33@Q�@�Q���33G�B�=q@Q�A,z����R�a�B��{                                    Bxe��v  �          A�z�?�@�������B�(�?�A-���(��b�
B�{                                    Bxe��  6          A�33@UA �����p�B��)@UA8������W\)B��q                                    Bxe���  r          A���AAQ����OBFQ�AAHQ��\  �*�BdG�                                    Bxe�h  �          A�
=@��HA z���Q��Sz�B^�H@��HAP���[\)�+��ByG�                                    Bxe�  �          A���@��HA���  �X\)Bb{@��HAQ��b�R�0G�B|��                                    Bxe�)�  �          A��H@��HA"�R�~�H�Q�B`G�@��HAR�\�Y��)Bz�                                    Bxe�8Z  T          A�  @�ffA��Q��Qz�BS�R@�ffAN{�[��*�RBo�
                                    Bxe�G   �          A��\@�{A#��~ff�N  BW�
@�{AS\)�X(��&�
Brz�                                    Bxe�U�  T          A�z�@�ffA4(��{��G��Bi33@�ffAc33�R{��
B��                                    Bxe�dL  �          A��
@޸RA@���o��<=qBp
=@޸RAmG��C���HB��R                                    Bxe�r�  �          A�ff@�G�AC\)�pz��?G�B~��@�G�Ap  �D  ���B�(�                                    Bxe���  �          A���@��ABff�t���C33B�\@��Ap  �Hz��{B��                                    Bxe��>  T          A�(�@�G�AF�\�o33�>{B�@�G�As
=�A���HB�G�                                    Bxe���  �          A�ff@���AF=q�v�H�BG�B��@���At(��Ip���HB�W
                                    Bxe���  
�          A�  @�p�AG��r=q�>z�B�G�@�p�At���Dz��=qB��R                                    Bxe��0  �          A�33@�G�AK��n{�;{B��@�G�Ax  �?���B�
=                                    Bxe���  �          A���@�(�AL���lz��:Q�B��{@�(�Ax���=��B�=q                                    Bxe��|  T          A��
@�  AE�u��D��B��@�  As��G\)���B�W
                                    Bxe��"  T          A���@��\AL���m�=�B���@��\Ay��>�H�B��=                                    Bxe���  
�          A��@��AmG��Q�!B�  @��A��������B��                                    Bxe�n  �          A�@��
Ap(��Mp����B��=@��
A������߅B��3                                    Bxe�  �          A�G�@�\)Aq��H(���B�#�@�\)A����=q��B�p�                                    Bxe�"�  �          A�G�@�\)A{33�<����\B�(�@�\)A�ff����
=B��q                                    Bxe�1`  �          A�z�@�p�A|z��7��
=B�\)@�p�A�z����R��z�B��                                    Bxe�@  �          A��@�G�A;��o33�IG�B�@�G�Ai��C33���B�
=                                    Bxe�N�  �          A��@qG�A^�\�Y���-Q�B���@qG�A�p��&�R��B�#�                                    Bxe�]R  �          A���@��AN�R�lz��=ffB�{@��A{\)�<z��z�B�L�                                    Bxe�k�  �          A�(�@��AW��aG��3(�B��q@��A����/���B�\)                                    Bxe�z�  �          A��
@��A^�H�U��'��B��=@��A�33�!���z�B��                                    Bxe��D  �          A��@�
=Al(��Q��!�B�L�@�
=A�\)�33��\)B���                                    Bxe���  �          A���@q�Ahz��V{�&�
B���@q�A�  � ����B��f                                    Bxe���  �          A��
@1G�Al���X(��'�\B�@1G�A�ff�!����B�Ǯ                                    Bxe��6  �          A���@`��Aj=q�N�\�"p�B�
=@`��A�(������=qB��3                                    Bxe���  �          A��R@���Aq��F{�ffB���@���A����R��G�B���                                    Bxe�҂  �          A���@�=qAxQ��9���
=B�(�@�=qA��H����\)B�aH                                    Bxe��(  �          A�z�@У�Ai�?\)�{B��)@У�A�ff�	����{B��\                                    Bxe���  �          A�G�@ȣ�Au�5p��	\)B��\@ȣ�A�G����\���B�\)                                    Bxe��t  �          A��@��A����'�
��  B��@��A�p��ٙ���=qB�L�                                    Bxe�  �          A��
@�\)A�(���\��  B��@�\)A�{��\)��\)B�G�                                    Bxe��  �          A��@�  A�p��$  ��  B���@�  A���
=��p�B��R                                    Bxe�*f  �          A��\@*=qA��H�z���z�B�{@*=qA�z����R��33B��3                                    Bxe�9  �          A��R@
�HA�\)�����B���@
�HA����������B��)                                    Bxe�G�  �          A�(�?�\)A�����=qB�aH?�\)A�\)��Q���  B��=                                    Bxe�VX  �          A�\)@�  Aa�>=q���Bx�@�  A��R�	p���ffB��3                                    Bxe�d�  �          A��R@��Al���3\)�	�B��@��A����������HB�ff                                    Bxe�s�  T          A��\@���A~{�'33��Q�B��
@���A�  ��G���(�B��3                                    Bxe��J  T          A�p�@��RA��
�
=��{B���@��RA�p���
=��33B��                                    Bxe���  �          A��@�ffA~�\�G���\)B�u�@�ffA�G������B�W
                                    Bxe���  �          A�G�@���A}�   ��
=B��@���A�33�ʏ\��z�B��3                                    Bxe��<  �          A��H@���A\)����z�B��
@���A��������\)B�p�                                    Bxe���  �          A���@�ffA�����\����B�ff@�ffA����ff��Q�B���                                    Bxe�ˈ  �          A��H@�ffA����{��p�B�k�@�ffA��
��G��n{B�.                                    Bxe��.  �          A�G�@��A�ff���̣�B�z�@��A������V�HB���                                    Bxe���  �          A�
=@�ffA���ff��ffB�Ǯ@�ffA�{���R�F=qB�\                                    Bxe��z  �          A�=q@��\A�
=�����\)B�z�@��\A�������?�B���                                    Bxe�   �          A���@���A���\)���B�k�@���A����G��b�\B��H                                    Bxe��  �          A�  @�{A�=q�=q��  B�Q�@�{A�
=�����|  B�                                      Bxe�#l  �          A�
=@�A�=q�ff�ޣ�B�.@�A�������zffB��                                    Bxe�2  �          A�z�@��A}��  ��  B�z�@��A�(�������33B�aH                                    Bxe�@�  �          A��@ə�A{��=q��p�B�\)@ə�A�z���{�|  B�                                      Bxe�O^  �          A��?�ffAuG��'\)��B�z�?�ffA�{�ٙ����B�                                    Bxe�^  �          A�ff?���Ay�� Q����B��3?���A�G���=q��G�B��\                                    Bxe�l�  �          A�@0��At���#��Q�B��=@0��A�����=q��\)B�                                    Bxe�{P  �          A���z�Aj=q�6=q���B��
��z�A�z����H����B��                                    Bxe���  �          A�{��Ap(��=G���
BĀ ��A�(���H��p�B�\)                                    Bxe���  �          A�Q��eAi��H(��G�BϮ�eA�  ��H�ٮB��
                                    Bxe��B  �          A�z��|��Ai���@���{B�G��|��A�p��\)�ϙ�B�B�                                    Bxe���  �          A�z��B�\A]��Dz��#��B����B�\A�����
B�8R                                    Bxe�Ď  �          A��׿�\)AS��O��0�\B�\��\)A|����\���B�u�                                    Bxe��4  �          A��\�p�ABff�Xz��>z�BȨ��p�An{�'
=�
=B��                                    Bxe���  �          A���Dz�A6=q�`���H  B�#��Dz�AdQ��1��(�B�B�                                    Bxe���  �          A�
=��Q�AG��Y�5ffB���Q�As��&�H�(�B�ff                                    Bxe��&  T          A����RA@���X���6��B�R���RAm��'��33B�p�                                    Bxe��  �          A�Q���{A?
=�Z�H�7�B����{Ak�
�)��\B�33                                    Bxe�r  �          A�����A<���^{�<{B�W
����Ajff�-��(�B�                                    Bxe�+  �          A�  ��p�A=��\���:G�B�
=��p�Ak\)�+\)�	ffB�p�                                    Bxe�9�  �          A�  ��{A7��]��:��B���{AeG��-��
��B�R                                    Bxe�Hd  �          A�
=��ffA@Q��[��4�RB�ff��ffAm���)����B��                                    Bxe�W
  T          A��\��  AB�\�ap��8ffB�z���  Aq��.�R�B�aH                                    Bxe�e�  �          A�z����AD���_
=�5�B����Ar�R�+��=qBߔ{                                    Bxe�tV  T          A�{��ffA:�R�d(��>�RB�\��ffAj=q�2�H�Bފ=                                    Bxe���  �          A���ffA8z��g\)�CQ�B䙚��ffAh���6�\��B�W
                                    Bxe���  �          A�\)���\A/\)�eG��Ep�B������\A_��6ff��B߅                                    Bxe��H  �          A�
=��
=A+
=�j�R�H��B�\)��
=A\���<z��Q�B�                                    Bxe���  �          A�����A/��pQ��J�B�.���Ab�R�@���ffB�=q                                    Bxe���  
�          A�z���
=A(��~{�Y�B�Q���
=AO\)�S��*�B�B�                                    Bxe��:  T          A��
����@������j(�B�.����A5��[��<=qB�
=                                    Bxe���  �          A����(�@�����u
=B�����(�A0Q��`Q��EG�B���                                    Bxe��  �          A�33���A�H�v�R�d=qB�
=���A9��Q��5ffB�R                                    Bxe��,  �          A�\)����@�\�w
=�iQ�B��{����A0  �S��;33B�aH                                    Bxe��  �          A�  ����@�=q�i���m=qC������AG��J=q�@p�B�                                    Bxe�x  �          A����ff@�Q��d  �l��B�  ��ffA&�R�B{�;�HB�=q                                    Bxe�$  �          AqG�?:�HADz��ָR����B�8R?:�HAXQ��g
=�n�HB��
                                    Bxe�2�  h          A��@0  AQ�8Q����RB�B�@0  A\)?�G�A��B�                                      Bxe�Aj  h          A���H�vff���H�Y\)Cm�H��H�\)��\{C^�q                                    Bxe�P  �          A���Q��Q��=qG�C��\��Q콏\)��H¯��C?33                                    Bxe�^�  �          A6�H�(������)����C��
�(���  �4z�£��Cw�q                                    Bxe�m\  �          A)��=L�������\�r\)C�J==L���4z��"{�)C���                                    Bxe�|  �          A9�*=q�x���(  �RCk���*=q��G��2{p�CML�                                    Bxe���  �          AG���{�����"ff�Y�HCe�R��{�;��2�H�~�
CT�                                    Bxe��N  �          AH  ��{������
�E��Cd���{�e��*�R�j\)CU\)                                    Bxe���  �          AK�
������ff�1Ca!H�����33�#
=�U�CT��                                    Bxe���  �          AIG���������"�RCS�3���:�H�33�;\)CG�\                                    Bxe��@  �          AO
=���Ϯ��\���CZJ=�����R��R�.�CPxR                                    Bxe���  �          AM���
{��  �ָR��CZ�
{��p��=q�ffCQ��                                    Bxe��  �          AO���H��Q���33��
=C_}q��H��\)� �����CX+�                                    Bxe��2  �          AK��z�����������C`\)�z���(����H�\)CZ��                                    Bxe���  �          AP  ����  ��\)��{C^5������
=���
����CYp�                                    Bxe�~  �          AR�H��z��`  �w33C^\)���p������p�CZff                                    Bxe�$  T          AQp��{�
�H�0���B�RC^�=�{���������ffC[��                                    Bxe�+�  T          ALz���
=��z����
Cf������p  ��G�Cd�                                    Bxe�:p  
�          AI���p��&{��
=��Ck\)��p��\)�>{�YCj:�                                    Bxe�I  �          A<z��У����L�;aG�Cln�У��  �{�A��Ck�\                                    Bxe�W�  �          A;������((�>k�?���CtW
�����#�
�Q��<z�Cs                                    Bxe�fb  �          A:�H���7
=��{��C������/�
�J�H�}C�U�                                    Bxe�u  �          A@�ÿ��>�R?
=q@#�
C��q���:�H����<  C���                                    Bxe���  �          AF{��{�AG�@Q�A�HC�4{��{�C��s33��z�C�@                                     Bxe��T  �          AD�Ϳ���?33@'�AD��C��=����C���ff��C��)                                    Bxe���  �          AEp���p��=�@E�Ag
=C��ÿ�p��C�<��
=��
C��
                                    Bxe���  �          A6ff��(��)G�@p��A��\C�b���(��2�H?z�H@��RC��H                                    Bxe��F  �          A6�R�/\)�!@�A�\)C~�{�/\)�.ff?޸RA�C�H                                    Bxe���  �          A3�
�=q�\)@�  A��C�0��=q�,Q�?�A�
C���                                    Bxe�ے  �          A.{�*�H���@�A�z�C~��*�H�#
=@�A;
=CT{                                    Bxe��8  �          A��   �Q�@l��A��HC����   ��R?�{A z�C��                                    Bxe���  �          Ap������@FffA�ffC��
����?c�
@�p�C�0�                                    Bxe��  �          A��p���@J�HA�G�C�\�p���
?Y��@�33C�J=                                    Bxe�*  �          A�H��{�  @?\)A�{C�(���{�\)?
=@a�C�p�                                    Bxe�$�  �          A�ÿ�p����@ ��AyC�p���p��ff>.{?���C��                                    Bxe�3v  �          A녿�z���@(��A�{C�Q쿔z��>�  ?��C�w
                                    Bxe�B  �          A녾�����@Tz�A�=qC�������z�?fff@�p�C�
                                    Bxe�P�  �          AzῬ���p�@�Al��C�Ǯ������\<#�
<�C��                                    Bxe�_h  �          A(��0  ��
@ffA`(�C}��0  ��׽L�;���C}��                                    Bxe�n  �          A!��G��=q@g
=A��C�
=�G��(�?�\)@���C�w
                                    Bxe�|�  �          A-��z�H�\)@dz�A��Cwz��z�H� ��?�  @��Cx��                                    Bxe��Z  �          A)�K��ff@qG�A���C{T{�K�� ��?��H@���C|xR                                    Bxe��   
�          A.�R�>�R�33@w
=A���C|�f�>�R�&{?��H@��
C}�3                                    Bxe���  �          A)��Q���\@�{A��
C�u��Q��"{@Q�AO�C��                                    Bxe��L  �          A"�H��33�{@�G�A���C�N��33�G�@33AQp�C���                                    Bxe���  �          A z�.{���@��B33C�k��.{��@L(�A���C�}q                                    Bxe�Ԙ  �          A녿�z��(�@�\)Aޏ\C�~���z��ff@	��AO\)C��                                    Bxe��>  �          Aff�����@��B
�C�0�����	G�@@��A�z�C�k�                                    Bxe���  �          Azᾅ����@�ffB�HC��������@`  A���C�&f                                    Bxe� �  �          A�H�����G�@��\B��C�lͿ���(�@aG�A�p�C�^�                                    Bxe�0  �          A�������
=@���B/C|�=������33@�\)A�33C��                                    Bxe��  �          Ap���������@�=qB6�C��ÿ�����\)@�  A�
=C��                                    Bxe�,|  �          A�R�����  @�Q�B;G�C��q������@�A��HC��                                    Bxe�;"  �          A33��
���
@�G�B7��Cv����
��G�@�z�A��C{33                                    Bxe�I�  �          A{�\�����@��\B33CoG��\�����H@u�A��Ct�                                    Bxe�Xn  �          A(��Q�����@�z�BC��Ct���Q����@��B
\)Cz                                    Bxe�g  �          A\)�����Ǯ@�  B#�C�׿�����@hQ�A�  C��q                                    Bxe�u�  �          A  ��z����R@�z�BO�C���z���@��B�C�k�                                    Bxe��`  �          A�H�������A��Bnz�C|�������@޸RB2
=C���                                    Bxe��  �          A z���R����AffB�CS�\��R�s33A
=By�Co�H                                    Bxe���  h          A��P���?\)A�Bu�C^���P����=q@�  BE�CmG�                                    Bxe��R  �          A�H�g
=��G�A�
Bb�Cd33�g
=��(�@��HB/z�CoxR                                    Bxe���  �          A�b�\��ff@��
BBffCl�R�b�\��  @��BG�Cs�3                                    Bxe�͞  �          A���mp��|(�@�G�B\�
Cb�R�mp����@�p�B*ffCm�f                                    Bxe��D  �          Ap������33@��B(��Ce@ �����z�@�ffA�\Cl@                                     Bxe���  �          A�
������\)@��A�Q�Cl�������Q�@-p�A�ffCo�q                                    Bxe���  �          Ap����\�Dz�@�p�B,\)CO�=���\����@��HB	33CZ��                                    Bxe�6  �          Az����@G�@��
B&(�C"B����>���@�ffB1��C1�                                    Bxe��  �          A  ��=q@�@�\)B{C ���=q?(��@�(�B&�C.�                                    Bxe�%�  �          A{����@�@�=qB(�C"޸����?^�R@�G�BQ�C-��                                    Bxe�4(  �          A$����@I��@���B
=C�3��?�{@�p�B"(�C)��                                    Bxe�B�  �          A!���@_\)@��B\)C����?�p�@�33B#C&޸                                    Bxe�Qt  �          A2{��@W�@��A���C����?�\)@У�B33C)z�                                    Bxe�`  �          A0����R@U�@��HA�G�C����R?��@�G�B
=C)�{                                    Bxe�n�  
�          A)���H��@<��A��
CFT{���u?��HA-��CJ\                                    Bxe�}f  �          A#33��H�mp�@
=A@(�CJ  ��H����?z�H@��\CLQ�                                    Bxe��  �          A0����?�33@�{A��\C+5�����  @��BffC5�
                                    Bxe���  �          A8Q���R@'
=@�z�BffC$)��R?5@��
B
=C/�
                                    Bxe��X  �          A?
=��@:=q@�=qA�=qC#�3��?�z�@��B=qC-c�                                    Bxe���  �          A;
=�p���@�A�
=C?���p��a�@�(�A���CG��                                    Bxe�Ƥ  �          AA���"�R�tz�@�p�A˅CH�)�"�R��(�@xQ�A�Q�CN�                                    Bxe��J  �          A<(��(��Q�@θRB�CB��(����@���A�ffCKaH                                    Bxe���  �          A9G������@��B  CB�������H@��A�{CK�                                    Bxe��  �          A9�)����@��
A��\C>@ �)���E�@h��A�  CD8R                                    Bxe�<  �          A:{�&{�˅@��
A��HC<�R�&{�:�H@��RA��\CC��                                    Bxe��  �          A@���+\)���@��A�  C=}q�+\)�J=q@��HA��CDu�                                    Bxe��  �          A4z���ÿ�=q@��
A�{C:L������@�33A�z�CA�                                    Bxe�-.  �          A6ff�
�H?�=q@�Q�B�RC,�f�
�H��ff@�Q�B�
C:�f                                    Bxe�;�  �          A5���@R�\@�{B�C����?�Q�@�z�B!ffC)ff                                    Bxe�Jz  �          A=� ����Q�@�\)B��CS��� �����@��AӅC\@                                     Bxe�Y   �          A>�\�=#�
@�G�B��C3�=��   @�  BC@{                                    Bxe�g�  �          A@��������
@���B#\)C=�
����n{@߮B(�CJ��                                    Bxe�vl  �          ADQ����p�@��BffCQ����׮@��
A��HC[�q                                    Bxe��  �          ADQ���
=��(�A	G�B2(�CV
��
=��p�@���BC`�\                                    Bxe���  �          A?
=�ҏ\�E�A��BO�
CM��ҏ\����@�\)B*z�C\��                                    Bxe��^  �          AF�\��R����@�Bz�CR�3��R���@�33A��C\:�                                    Bxe��  �          AF�\� Q����A�B>�
C4�q� Q��+�AffB3=qCF�                                     Bxe���  h          AC�
����?�RA�B[33C.�{������\A�BS�CFY�                                    Bxe��P  �          AG����H���A
=B<z�CSB����H��  @��Bp�C_��                                    Bxe���  �          AJ=q��ff���@�Q�A���Cn����ff�-G�?�33A�Cq8R                                    Bxe��  �          AK33������@�Q�A���Cg������$��?��A��Cj                                    Bxe��B  �          ALz���p��  @�G�AׅCj8R��p��(��@2�\AICm�R                                    Bxe��  �          AL����G�� ��@�ffA��Cbٚ��G����@qG�A�G�Cg�
                                    Bxe��  �          AF{����׮@�ffB�
Cbu�����
�\@��A�
=Ciz�                                    Bxe�&4  �          AG33�������A�BF�HCjxR������@��HBQ�Cr                                    Bxe�4�  �          A?���Q���33A��BT(�C](���Q���p�@�B p�Ci�R                                    Bxe�C�  T          AE��
=�o\)A��BR�RCR��
=��Q�A�
B'��Ca&f                                    Bxe�R&  �          AG�
���\����A�RBR�HC[n���\���A Q�B 33ChQ�                                    Bxe�`�  �          AL���Ӆ��AG�BC�CZ\�Ӆ��=q@�\B{Ce�=                                    Bxe�or  �          AL����=q��p�A��BGG�CVc���=q����@�p�B�
CcW
                                    Bxe�~  �          AIp���\���RA  B7��CT�R��\��R@���BG�C`��                                    Bxe���  �          AB�H���H��{A z�B_�CZ&f���H����A��B,��Ch�\                                    Bxe��d  �          AD  ������A�B;��Cd=q�����	G�@ҏ\B33Cm@                                     Bxe��
  �          AA��������@�
=B�Cr������#�@�=qA�p�CwY�                                    Bxe���  �          A=���z���
@�(�B��Cl����z����@c�
A�G�Cq(�                                    Bxe��V  �          A<(��n{�(�@��
B��Cu�n{�\)@���A�Cy��                                    Bxe���  �          A:{������z�@�G�B�HCfn�����{@���A���ClaH                                    Bxe��  �          A8Q����
����@ƸRB�C]�����
�   @|��A��CdW
                                    Bxe��H  �          A:�R������z�@��B�HC[}q������\)@�Q�A���Cc=q                                    Bxe��  T          A8����\)�ҏ\@��
B�C_L���\)�=q@r�\A��
Cec�                                    Bxe��  �          A8����@ff@���A�ffC&p���?��@�  A�{C0s3                                    Bxe�:  �          Ah���A/�
>�z�?��Ch��A%@k�Ak�
C{                                    Bxe�-�  |          Aq�(�A=G�?�  @s33C \�(�A/
=@��
A���CB�                                    Bxe�<�  �          Aqp���A1��@$z�AG�CxR��A(�@�z�A�z�C+�                                    Bxe�K,  h          As���RA@Q����
=B�(���RA9��@N{AF�\B�#�                                    Bxe�Y�  �          Ap(���AO
=���
�ۅB�p���AM��@AQ�B�Ǯ                                    Bxe�hx  �          AhQ���G�AL�ÿk��h��B��
��G�AG\)@HQ�AG
=B�.                                    Bxe�w  �          Ab�\����AEG������B�������AFff?�\@�RB�\                                    Bxe���  �          A`����p�AF�R�{�G�B��
��p�AG�
?�ff@��HB�{                                    Bxe��j  �          Ac
=���
ALQ��\(��`��B߽q���
AS33?@  @B�\B�k�                                    Bxe��  �          Ae����AP  �C�
�F{Bޔ{���AT��?�Q�@�G�Bݮ                                    Bxe���  
�          A^ff�
=A+�
?�=q@�Q�C ���
=AQ�@��A���C�H                                    Bxe��\  �          A]p��A!�?L��@S�
C��A��@}p�A�\)C
=                                    Bxe��  �          A\z��\)A,�ÿ��
�HC ��\)A&�\@;�AEp�C�                                    Bxe�ݨ  �          A]����
=A4�׿k��r�\B�p���
=A/�
@.{A5G�B��H                                    Bxe��N  �          A^ff�33A"{@��ACB��33Aff@��A�(�C�3                                    Bxe���  �          AV=q�Az�@i��A}p�C�q�@�@�33A��C��                                    Bxe�	�  �          AU���%G�@���@�p�A��CT{�%G�@��@�
=A�RC��                                    Bxe�@  �          AO33��R@љ�@��HA��C.��R@�=q@�p�B�C�                                    Bxe�&�  ,          AN�R���A1p�������HB�B����A.�R@A)B�                                    Bxe�5�  �          AW��ӅA:�H������B����ӅA6�\@.{A:�RB�#�                                    Bxe�D2  �          AT����p�A5�B�\�QG�B�R��p�A0  @<(�ALQ�B�aH                                    Bxe�R�  �          AO\)���RA8Q쿀  ���RB�q���RA3�@1G�AF�RB��                                    Bxe�a~  ,          AU����\A:�H����33B�
=���\A:�\@   A�
B��                                    Bxe�p$  T          A]��  AA��Z�H�d  B��H��  AHQ�?.{@333B�33                                    Bxe�~�  h          AW33��  A9G�� ���G�B��
��  A9��?�\)A33B�q                                    Bxe��p  �          AS\)��\AJ{�c�
�}p�Bȅ��\AC�@QG�Al  B�.                                    Bxe��  h          AS�����A<z�?��@���B�33����A+33@��HA�33B�=                                    Bxe���  ,          AX����(�A<Q�=#�
>#�
B�(���(�A1@xQ�A��\B�Ǯ                                    Bxe��b  �          AYG���G�AEp��mp��}B�\)��G�AM�?
=q@G�B��H                                    Bxe��  �          A^{��p�AE���`  �j=qB�p���p�AM�?B�\@HQ�B��                                    Bxe�֮  �          AW
=��z�A7�=�Q�>���B�8R��z�A,��@w�A���B�33                                    Bxe��T  8          A[
=��Q�A@Q쿦ff��\)B�
=��Q�A<z�@/\)A9��B�                                      Bxe���  �          A\(����HA,��@=p�AI�B�����HA�@��
A�ffCn                                    Bxe��  �          Ak33�%G�Aff@�A�\)C
z��%G�@�  @�{A���C�f                                    Bxe�F  �          Ak\)�)�@�
=@���A���C��)�@�\)A
=B  C�3                                    Bxe��  �          AuG���HA;
=@fffAYC #���HAG�@���A�p�C�                                    Bxe�.�  �          Ax����AL��@p�AB���A4  @��HA�z�B�                                      Bxe�=8  �          At���AJ=q@��A�
B���A2�\@�33A�  B���                                    Bxe�K�  �          Ao�
���AJff?�=q@���B�q���A6=q@�
=A���B�8R                                    Bxe�Z�  �          AlQ��G�A@  @33A   B��\�G�A)�@��RA�z�C !H                                    Bxe�i*  |          Aj�H��{AG33>u?n{B�k���{A:ff@��A�(�B��f                                    Bxe�w�  T          Af{��(�A<  @�\A
=B����(�A$��@ÅA��
B��)                                    Bxe��v  �          Ab�\�A6{?��@{B����A(z�@�33A�p�C s3                                    Bxe��  �          A_
=� ��A4��?��
@���B��
� ��A#\)@���A��C 8R                                    Bxe���  �          A]���A0z�?�{@ָRB��R��Ap�@�  A���C��                                    Bxe��h  |          A_\)��z�A5p�?�
=A{B�.��z�A Q�@�p�A��
B��)                                    Bxe��  �          AX����G�A=�@z�A��B�����G�A'\)@�\)A���B�\                                    Bxe�ϴ  �          ATz����AC�?�=q@�33B�W
���A/\)@���A��B���                                    Bxe��Z  �          A[�
���A=?�p�@���B��)���A(��@�p�A�Q�B�                                    Bxe��   �          A[�
�(�A+33?J=q@R�\C � �(�A��@���A��
C�                                    Bxe���  �          A\(���
A+�?�G�@�=qC aH��
A��@��HA��
C�H                                    Bxe�
L  �          A]�����RA4��?�A (�B��=���RA\)@�{A�z�B�u�                                    Bxe��  �          A]��HA'33?�p�@�(�C���HA��@�\)A��HC��                                    Bxe�'�  �          A^ff��Q�A8��?�\@��B�\��Q�A#�
@�z�A���B��=                                    Bxe�6>  �          Ai��
�\A9?�=q@�  B�k��
�\A'
=@�Q�A��C��                                    Bxe�D�  �          Ac\)� ��A;33?E�@G�B�
=� ��A+\)@���A��HB��
                                    Bxe�S�  �          Ac33����AF=q?��
@��B�\)����A4Q�@�Q�A�Q�B�\                                    Bxe�b0  �          A`(���z�AH��?+�@0  B����z�A8��@��A��B�                                    Bxe�p�  �          A`������AM�=�>�B㙚����A?�@�=qA�=qB�                                    Bxe�|  |          Ad����z�AT  ���R��G�B�=q��z�AHz�@�=qA��RB�                                    Bxe��"  �          Ap  ��ffAd  ����Q�Bֳ3��ffAX��@�ffA���B�aH                                    Bxe���  �          Aj=q��33A]녿�Q���z�B�����33AYG�@QG�ANffB׀                                     Bxe��n  �          Ac\)��
=AM���<���A�B�L���
=AQ�?޸R@�Bݨ�                                    Bxe��  �          AlQ���ffAU�J=q�Dz�B���ffAL��@|��Ax��B�3                                    Bxe�Ⱥ  �          Ak�
���HAV�R�(���B������HALz�@���A��RB���                                    Bxe��`  �          Ad������AO�
�#�
�#�
B��)����AF{@~�RA��B���                                    Bxe��  �          Ak
=����AQp�>�{?�=qB��H����AB{@�{A�G�B�                                     Bxe���  �          Ag33�ҏ\AJ�R��
=��
=B��
�ҏ\A@  @��HA�33B�p�                                    Bxe�R  �          Ab�\��p�A@z��L(��R�\B����p�AF{?��
@�Q�B�Ǯ                                    Bxe��  �          Ab{����A1p���Q����
B��
����AEG����H��B홚                                    Bxe� �  �          Aap���G�A7���  ��33B�8R��G�AEG�����\B��3                                    Bxe�/D  �          A]����A;\)�Z=q�dz�B�Q�����AB�\?z�H@��\B�                                     Bxe�=�  �          A\����RA1G��P���[�B��f��RA8Q�?h��@q�B��)                                    Bxe�L�  �          A]��RA,Q�����C ���RA-�?�(�@�z�B��=                                    Bxe�[6  |          A\��� (�A.�R�=q�!p�B��� (�A0��?�z�@�ffB���                                    Bxe�i�  �          A]����A?����\��\)B�R����AJff>��H@33B�ff                                    Bxe�x�  �          A]�����
AA�� ���(  B�R���
AB�R@   A�B�\)                                    Bxe��(  �          A[\)��(�AA�p��%p�B�Ǯ��(�AB�H@z�A\)B�                                    Bxe���  �          AV�H��Q�A=녿���Q�B�3��Q�A;�@%A2=qB�G�                                    Bxe��t  �          AUG�����A>{��!�B�������A>�H@�A�\B��                                    Bxe��  �          AUp����HA=��>{�NffB䞸���HAAp�?�G�@���B��                                    Bxe���  �          AYp���
=AD  �;��H  B�.��
=AG�?��H@��B�p�                                    Bxe��f  �          A]p�����ALz��1G��8��B�������AN�R@z�A	G�Bڙ�                                    Bxe��  �          AZ�\�Dz�AM�W
=�dz�B��f�Dz�AS33?�ff@���B�8R                                    Bxe���  �          A\���QG�AI���G����B��QG�AV�\>\?�=qB�ff                                    Bxe��X  �          AZff��z�AI���1��<Q�B�ff��z�AK�
@G�A(�B�                                      Bxe�
�  �          AW33���\A@�ÿ�p���B����\A?33@$z�A0��B�{                                    Bxe��  �          AT  ��  A;��:=q�M��B�8R��  A?�?���@��B�\)                                    Bxe�(J  �          AP�����A��@z�A#�
C� ���@�33@�ffA�(�C                                      Bxe�6�  �          AS��&�\@�p�?�33@�{C���&�\@�Q�@�ffA��
C�                                    Bxe�E�  �          A@(��=q@z�����{C(.�=q@o\)�tz����HCG�                                    Bxe�T<  �          A5��\@
�H����|Q�C%����\@:�H��p���HC �)                                    Bxe�b�  �          A=�'
=@�  ?J=q@s�
C:��'
=@�=q@0��AXQ�C:�                                    Bxe�q�  �          ADQ��(��@��
?�\)A��C���(��@�Q�@z=qA�p�C��                                    Bxe��.  �          AEp��(��@�  @�AffCT{�(��@�G�@�{A�p�C��                                    Bxe���  �          AH���/�@��
?�\AG�C���/�@�G�@tz�A���Cs3                                    Bxe��z  �          AH���-@���@ ��A�
Cu��-@�\)@�=qA���Cu�                                    Bxe��   �          AI���%p�@���@�A  CO\�%p�@��
@�  A��Cz�                                    Bxe���  �          AC\)�)�@�\)@ ��AA��C��)�@k�@�{A�  C ��                                    Bxe��l  �          AH���33@陚@8��AV�\C
=�33@�@�p�A��
C�H                                    Bxe��  �          AV�R�$��A\)@(�A'�C}q�$��@�{@��HA�C                                    Bxe��  �          AQ�'
=@�p�@L��Ab�\C���'
=@�{@��A�33Cz�                                    Bxe��^  �          AO\)�#�@�{@L(�Ad��C���#�@��R@��A�(�C�                                    Bxe�  �          AHQ����RA�@8��AU��C�\���R@�ff@���A뙚C�f                                    Bxe��  �          AFff��ffA(�@p��A�B�\)��ff@�(�@�\B�CJ=                                    Bxe�!P  �          AH(����A��@i��A��CǮ���@߮@��B  C
#�                                    Bxe�/�  �          AK
=�>=q@���?��@�{C E�>=q@\��@5�AN{C#�{                                    Bxe�>�  �          A8Q��%G���@%A]��C=���%G��*=q?ٙ�A�CBp�                                    Bxe�MB  �          A<����
���@�AC�
CVW
��
��Q�#�
�J=qCX0�                                    Bxe�[�  �          A=��
�H��\)@'�AP��C[�=�
�H��{���ÿ��C]��                                    Bxe�j�  �          AM��7�
@9��@��A�33C%�)�7�
?���@��A�z�C.ff                                    Bxe�y4  �          AL(��;
=?�@�
=A��HC-
�;
=�   @��A�  C6p�                                    Bxe���  �          AK��4Q�@Vff@���A��HC#u��4Q�?�(�@��A���C-�
                                    Bxe���  �          AIG��-=��
@��RA���C3�\�-���@z�HA�\)C<�                                    Bxe��&  �          ALQ��>�H��@s�
A���C6�=�>�H����@Tz�Aup�C=L�                                    Bxe���  �          AF{���@�{@��A�{C�R���?��@��
B�
C)5�                                    Bxe��r  �          AH(��0z��E�@�{A���CC���0z�����@:=qA[�CJn                                    Bxe��  �          A@���.�H>.{@�A�=qC3��.�H��  @��A�
=C=�                                    Bxe�߾  �          AG��7\)?��
@�33A�(�C-�)�7\)���@��A���C6�q                                    Bxe��d  �          AL���*�R@�(�@��RA�Q�C�)�*�R?˅@��
B�C+�                                    Bxe��
  �          AL  �4(�@#�
@�\)A���C'0��4(�>�(�@��\A���C1�{                                    Bxe��  �          AK��((�@���@��
A�p�C@ �((�@E�@��A�C#��                                    Bxe�V  �          AK��'\)@�@�{A��C��'\)@5@�=qA���C$�=                                    Bxe�(�  �          AO33���A33@�\)A��Ch����@�z�Ap�B!��C�                                    Bxe�7�  |          AN{��=qA*�\@��A�ffB�aH��=q@��A�B1�B�                                    Bxe�FH  �          AL�����A,��@��A���B������A ��A
ffB,Q�B�\                                    Bxe�T�  �          AN{����A!@��A�B�=q����@�=qA	�B,�B�                                      Bxe�c�  �          AT  ���RA1p�@p��A�\)B�ff���RA�@��BB�
=                                    Bxe�r:  �          AS
=�XQ�A4(�@��A�{B�p��XQ�AffAp�B=p�B�{                                    Bxe���  @          AO33��A
=@���A�B��=��@�z�@���B�HC
=                                    Bxe���  h          AS
=��@�\)@�B
C�q��@Q�A�\B:�CB�                                    Bxe��,  |          AW
=���A�H@�=qA�33B�����@�Az�B#C	��                                    Bxe���  �          Aa����Q�AT��@Q�A\)BՊ=��Q�A6{@�
=A��B���                                    Bxe��x  �          AV=q��z�A��@�ffB��B�����z�@���A)��BY33C��                                    Bxe��  �          A[
=��33AA��@z=qA�B߮��33A��A=qBG�B���                                    Bxe���  �          A[�
��(�A=@mp�A|z�B�����(�A�\AB  B��
                                    Bxe��j  �          Ac
=��  AO33?��@�B�{��  A2�R@�G�A��HB�z�                                    Bxe��  �          Ae����AO�?O\)@O\)B�=���A9p�@�p�A�33B�                                    Bxe��  �          A^ff�ٙ�A?�
?�=q@�  B���ٙ�A'�@�  A�p�B�
=                                    Bxe�\  �          A\z���=qA8��?���@љ�B����=qA�@��A��HB��{                                    Bxe�"  �          AY��(�A>{?�@��B� ��(�A#�@�G�A�ffB���                                    Bxe�0�  |          Ahz�����AQ��  �}p�B�=q����AA@���A�z�B���                                    Bxe�?N  �          Aa���z�AI�����  B�  ��z�AF�R@FffAK�B�R                                    Bxe�M�  �          Ad  ��33AMp����
��{B�  ��33AB�H@�
=A��RB�W
                                    Bxe�\�  �          A^�H��(�AHQ��\���f=qBޞ���(�AM��?��@��Bݞ�                                    Bxe�k@  �          Ah����G�AQ���qG��pQ�B����G�AX  ?�@�Q�B��                                    Bxe�y�  �          Ai����AR=q�p  �o
=B��)����AX��?�\)@�z�Bܽq                                    Bxe���  �          Aa�����AMG�����!��B��
����AK�@<(�AB=qB�#�                                    Bxe��2  �          Ab=q���\AQG�������B܀ ���\AK�@e�Al(�Bݏ\                                    Bxe���  �          Abff��Q�AO�
��
=��p�B����Q�A@��@�(�A�
=B�33                                    Bxe��~  �          Aa���ffAM���ff����B��f��ffAB=q@�  A���B�B�                                    Bxe��$  �          Ad����33AS�
>Ǯ?���B�  ��33A>�H@��A��B�G�                                    Bxe���  �          Aap���
=AP(�����  B���
=AA@���A��B�                                    Bxe��p  �          Ac33��AXz�>���?���B�\)��AC\)@�=qA�Q�B���                                    Bxe��  �          AXz���  A7\)@   A	�B����  A=q@�Q�A��
B��                                    Bxe���  �          AM��	Aff@�A�(�C�{�	@���@�33Bp�CT{                                    Bxe�b  �          AK33�p�@��@i��A��
CW
�p�@�=q@�  A��HC33                                    Bxe�  �          AM��@��@_\)A|(�C��@�{@�(�A�RC=q                                    Bxe�)�  �          AH���@�\)@%�A>ffC�@���@�=qAə�C�R                                    Bxe�8T  �          AH(��#�
@��@�
=A�z�C��#�
@ ��@��HA��
C&@                                     Bxe�F�  �          AM��@�p�@l(�A��\C
=��@�
=@У�A�=qC�                                    Bxe�U�  �          AIG���H@�@�  A��\C����H@���@�BG�C��                                    Bxe�dF  �          A@z���R@�=q@���A���C=q��R@(�@ǮA�z�C'�=                                    Bxe�r�  �          AE���33@�G�?W
=@y��C}q�33@�  @~�RA��CxR                                    Bxe���  �          AG33�
=Aff@�A(�C�R�
=@���@��
A��
C                                    Bxe��8  �          AH���p�A=q?���A�C&f�p�@��H@�33A͙�C
�f                                    Bxe���  �          ALz�����A!p��L�Ϳc�
B�G�����A(�@�Q�A���C
=                                    Bxe���  �          AJ=q��
=A (�?W
=@vffB�u���
=A(�@�A��Cz�                                    Bxe��*  �          AJ�H��G�A
=@�A&{C �=��G�@��@ÅA�33C��                                    Bxe���  {          AK����
A"�H@>�RAZ�\B�{���
A   @޸RB��C��                                    Bxe��v  �          AN=q��G�A((�@|��A��B��
��G�@�z�@�\)B\)C �\                                    Bxe��  �          AMG���ffA)�@p  A��B�����ffA ��@��\B�B�Q�                                    Bxe���  �          AM���G�A,z�@
=A+33B��\��G�A��@��A��HB�#�                                    Bxe�h  �          AO33��G�A2�\?�  @�B�Ǯ��G�A  @�G�A߅B���                                    Bxe�  T          AO�
�ə�A1p�@G�A!G�B�33�ə�A@�
=A��
B�Q�                                    Bxe�"�  �          AO
=��
=A*=q@%�A9��B�\��
=A	�@�=qB =qC �                                    Bxe�1Z  �          AN�H��ffA(  @ ��A�B��f��ffA
�H@�  A��
C�3                                    Bxe�@   �          AK�
��ffA(  ?��AG�B�����ffAQ�@���A�G�C ff                                    Bxe�N�  �          AJ{����A!@�A-��B��=����A�\@�p�A���C8R                                   Bxe�]L              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxe�k�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxe�z�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxe��>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxe���  
C          ADz��33AG�@@��Ac�
C	!H�33@���@��A�CE                                    Bxe���  
�          AD���Q�@���@e�A��
C
k��Q�@���@љ�B33C��                                    Bxe��0  �          AEG���@׮@���A�z�C�
��@�z�@�Q�A�\)C.                                    Bxe���  "          AAp��Q�@�(�@�{A��C�)�Q�@Mp�@�\)BC �f                                    Bxe��|  "          AAG��(�@�Q�@�p�A��\C��(�@�\)@�{BQ�C�=                                    Bxe��"  	�          A?�
�p�@���@���A��
CO\�p�@J�H@���B	��C!=q                                    Bxe���  �          A@z��	�@޸R@�
=A�C{�	�@��
@��B��C{                                    Bxe��n  T          A?�
��{A
�\@;�Ac33C����{@��@˅B ��C�=                                    Bxe�  �          A@  ��{Aff@aG�A�
=C ���{@У�@�  BG�C	�                                    Bxe��  T          ABff��A	�@p��A�ffC����@�p�@�33B33C:�                                    Bxe�*`  
�          A=G���RA��@.�RAW�C.��R@��@���BQ�C�f                                    Bxe�9  �          A6�\��R?���@��
A��C-T{��R��33@��A�
=C<=q                                    Bxe�G�  "          A733��H?�33@�  AۅC)#���H���@��A陚C7(�                                    Bxe�VR  �          A8���#
=@l(�@~{A�
=C 
�#
=?�{@�A�33C+�                                    Bxe�d�  "          A733�#�@3�
@���A��\C$���#�?+�@��HA�\)C0=q                                    Bxe�s�  
�          A6�\�!?�ff@�=qA���C+E�!�L��@��Aڏ\C8�                                     Bxe��D  =          A6�\�p�?#�
@���B	�C0��p����@���B �HCAE                                    Bxe���  �          A5��,��@��A�CC�����  @i��A��\CNO\                                    Bxe���  "          A8���ff����@�33A�  C>�f�ff��@�\)A�ffCKc�                                    Bxe��6  "          A6�H��H�Mp�@�\)A�  CGL���H��z�@p��A��CRh�                                    Bxe���  �          A9��\)�x��@�p�A�p�CI�
�\)��z�@.�RAZ�RCR+�                                    Bxe�˂  
�          A9���=q�N�R@�ffAᙚCF���=q����@`  A�G�CP�R                                    Bxe��(  �          A733���n{@�G�A�\)CI#�����{@,(�AZ�RCQz�                                    Bxe���  T          A:{�33�W
=@�G�B  CI)�33����@�{A���CU��                                    Bxe��t  �          A<Q���\)�+�AG�B8p�C8�=��\)��  @�{B
=CP�                                    Bxe�  �          A<z��\)��G�@�G�B&p�C:�H�\)��=q@��B��CNc�                                    Bxe��  �          A=���\���@�{B-=qC6����\�vff@�ffB�CL��                                    Bxe�#f  �          A4���녿�33@�{B�C@�������@��RAә�CO�                                    Bxe�2  
�          A(���ڏ\@�z�@�p�A��HC^��ڏ\@�@�G�B.(�C"�q                                    Bxe�@�  T          A8zῠ  �+\)@tz�A�=qC��Ϳ�  �4�Ϳ��
��p�C�ٚ                                    Bxe�OX  9          A<���P  �  @ӅB��Cy)�P  �(��@(�A3\)C|��                                    Bxe�]�            A7�
��ff��ff@���B.p�Cg)��ff�Q�@��A��Cp�3                                    Bxe�l�  �          A6�R��Q���Q�AB^�Cd8R��Q���\@�G�B�HCs#�                                    Bxe�{J  �          A=����Q���Q�@���B$�CZ(���Q��{@�G�A��RCf�                                    Bxe���  
Z          A=G�����  A	p�B:�\C\T{���(�@�(�AظRCj                                      Bxe���  "          A;�
������ff@���B�RC`�������(�@G
=Av�\Ci\                                    Bxe��<  h          A?\)�Q쿈��@�  B��C;+��Q��g�@��A�G�CK�                                    Bxe���  |          AZ�\�333AR�H�L���X��B����333AB�\@�z�A�
=B��f                                    Bxe�Ĉ  
�          A^�R��Q�AXQ���R�%G�B�8R��Q�AS\)@r�\A}B\                                    Bxe��.  
�          AY�����
AK33>���?��HB�  ���
A333@�Q�Aԏ\Bޮ                                    Bxe���            AW��7�AR{?z�@{Ḅ��7�A8  @��
A�{B�                                      Bxe��z  "          A[��Y��ATz��G���ffB��Y��A>�H@��\A�33B��
                                    Bxe��   �          A]��
=AXQ�?�
=@�p�B��
=A9�@��HA��B�                                      Bxe��  �          A[��Q�AVff?s33@\)B���Q�A8��@��HA�RB�=q                                    Bxe�l  �          A^�R�[�AU�?��\@�  B����[�A6=q@�A��BՔ{                                    Bxe�+  �          A^�R�L��AW
=?�ff@���B����L��A6�H@�A��B�L�                                    Bxe�9�  �          A]��{AW�?@  @J=qBƮ�{A;33@ָRA�33B�p�                                    Bxe�H^  �          AZ�\�`  AR�\���
����B��`  A<Q�@�z�A��HB�{                                    Bxe�W  �          AV�\�333AP  ���ÿ�33B�Q��333A<(�@���A�33B���                                    Bxe�e�  
�          Ab�R��33A_33?�
=@���BÅ��33A=�@�B  B�G�                                    Bxe�tP  T          As
=�G�Ak�?��\@��B���G�AI�@�Q�A�G�B��H                                    Bxe���  �          Atz��ffAk�
@E�A:ffB���ffA<Q�A=qB(�Bʔ{                                    Bxe���  T          Ar�H�ffAjff@B�\A9��B�Q��ffA;33A�B\)B�\)                                    Bxe��B  �          Al��>�ffAdz�@x��As�B�aH>�ffA/�
A�RB((�B�L�                                    Bxe���  "          Aup�?��Ak�@�G�Atz�B��
?��A4��A$(�B({B�Ǯ                                    Bxe���  �          Aw33���Ar�R@
�HA{B�� ���AH��A��B  B�p�                                    Bxe��4  �          Ak33�aG�Ac�
�B�\�8Q�B��aG�ALz�@���A��HB���                                    Bxe���  �          Ag
=�\)Ab�H>��
?��\B����\)AG�
@ָRA�=qBʔ{                                    Bxe��  �          Ap���J�HAg�?�ff@�B̳3�J�HAA�A�RB��B�ff                                    Bxe��&  T          Ar=q��
=Ad��@@��B��f��
=A<z�A��B�HB�ff                                    Bxe��  T          ApQ���p�A_
=?�G�@��Bܳ3��p�A;�@�A��B�{                                    Bxe�r  "          Ak��h��A`(�>�G�?�p�B���h��ADz�@أ�A���B�                                    Bxe�$  
�          Al�ÿ.{Ajff?���@�z�B�Q�.{AF�\@���B��B�G�                                    Bxe�2�  T          Ar�H>uAp(��8Q��0��B�#�>uA[
=@�ffA���B���                                    Bxe�Ad  9          Ao�
�!�Ag�?�  @ٙ�B����!�AAp�A�RB��B˞�                                    Bxe�P
  �          An�R���A^{<#�
<�B�8R���AEp�@��HA�  B��                                    Bxe�^�  
�          Ai��x��A\(������Bә��x��AU@��A��Bԅ                                    Bxe�mV  
�          Ak33��33AZ�H?�p�@��\B��H��33A9��@�A�33B�p�                                    Bxe�{�  "          Aj�R�%Ae�?Y��@Tz�B�p��%AF=q@��HA���B˞�                                    Bxe���  �          Ah���E�Aa����Q���ffḄ��E�AS
=@�A��B�G�                                    Bxe��H            Afff�B�\A]�����G�B����B�\AV{@�
=A���B͙�                                    Bxe���  	�          AW�
�˅A3
=��ff�  B�33�˅AT  ��G���(�B��3                                    Bxe���  �          AK����@�\?��RA5��B�u����@�33@��\B�B�                                    Bxe��:  
(          AQ����@��
A#
=BM��C�����?�A<  B�C.��                                    Bxe���  "          AW
=��ffA#�@�Q�A�\)B�z���ff@�\)AQ�B   C�{                                    Bxe��  �          APz�����A/�@�A\)B�aH����A��@��B \)B�8R                                    Bxe��,  
5          Ac33��  AI�?}p�@~�RB����  A+33@�p�A��\B�                                    Bxe���  
u          A]p�����A?
=�
=�p�B�aH����A8��@h��Av�RB��                                    Bxe�x  T          A[\)��
=AH�׿�����G�B�8R��
=A;�@�(�A��\B�                                      Bxe�  r          A[���A4zᾔz῝p�B�#���A"=q@��RA��B��f                                    Bxe�+�  	`          A^ff�{A-p�?�
=@�  B�ff�{AG�@�Q�A�
=C�                                     Bxe�:j  �          AY��Q�A	G�@}p�A��C	8R�Q�@���@�\)B	��C�                                    Bxe�I  �          AV{��@�\@�  A�z�C��@P��A��B*  C �R                                    Bxe�W�  �          A]��33@��RA�\BffC+��33?�33A!��B=�
C,ٚ                                    Bxe�f\  |          AX����z�@��A��B6�CY���z�=#�
A,��BWffC3�                                    Bxe�u  �          AW��G�@���A(�B4\)Cp��G����A(z�BNz�C6��                                    Bxe���  �          AYp���@G�A&{BJ�C� ���z�A*=qBQ�CB�)                                    Bxe��N  G          AR�\��z�?�Q�A#
=BQ�C%���z��A�A�RBIp�CI��                                    Bxe���  h          AS
=�\)?�\)AG�BG��C,5��\)�eA�B7\)CK�H                                    Bxe���  T          AW�
��?J=qA,��B\ffC.�����HA�RBC�CR
                                    Bxe��@  �          AN�H�{@�  @�p�A���Cc��{@   @�
=B�HC(J=                                    Bxe���  �          AU�z�A@A#33C���z�@޸R@ʏ\A���C�                                    Bxe�ی  |          AQ������@$z�A0��Br��C������8��A/\)BoCO�                                    Bxe��2  
�          AS�
��R@��@\(�A��\C#���R@���@��A�p�C�3                                    Bxe���  �          AS\)�J=q@h��>W
=?n{C#���J=q@G�?�33A��C&!H                                    Bxe�~  �          AS��;33@�{?�G�@���C
�;33@���@k�A�C�\                                    Bxe�$  �          AUp��<Q�@��׾�33��G�C��<Q�@�\)@ ��A/\)C�                                    Bxe�$�  �          AT���#�@��R@,��A<z�C
�#�@�Q�@��
A�ffC�
                                    Bxe�3p  �          AU���G�Ap�?�{@�{C���G�@�@���A�
=C@                                     Bxe�B  ,          A[����@�{A�BHffC�����>��A$Q�BzffC/�H                                    Bxe�P�  T          AW�
�ٙ�@�\A333BeQ�C!h��ٙ��P  A/\)B]�HCM��                                    Bxe�_b  |          AU���(�?�(�A*=qBW\)C)
��(��mp�A!G�BG��CM��                                    Bxe�n  �          AZ�\���H@��A/\)BUQ�B�����H?���AO�
B�  C%Y�                                    Bxe�|�  �          A\����@�z�A;\)Bi�Bգ���?\)AY�B��C%\                                    Bxe��T  �          A^=q�#�
A.�RA�
B  B����#�
@��AF�RB��B���                                    Bxe���  �          AV�H��(�A�A33B �RB����(�@u�A:{BzCaH                                    Bxe���  �          AP���-@���@��A��
C� �-?��\@���B�HC.�)                                    Bxe��F  �          AP��� (�@�33@��A�  C��� (�@4z�@���BQ�C$B�                                    Bxe���  �          AQp��#�
@��R@�G�A�Q�C�)�#�
@=q@��HB�C&�                                    Bxe�Ԓ  h          AT(��	@�?У�A(�C��	@�33@��AʸRC�                                    Bxe��8  �          APQ��1G�@��;u���C���1G�@���@,��AD��CxR                                    Bxe���  h          APQ��θRA'\)�`  �~�\B�aH�θRA-�?�\)A�HB�k�                                    Bxe� �  �          AS\)���A=q?�p�@���Ck����@��@�  A�=qC+�                                    Bxe�*  �          AV�H�(�A�
@*=qA7�
C���(�@���@ӅA�G�C�                                    Bxe��  �          AM�7�
@�
=@1G�AG\)C�
�7�
@C�
@���A�C%�                                    Bxe�,v  �          AO
=�8(�@��@H��AbffC���8(�@+�@�Q�A��C&��                                    Bxe�;  �          AQ���#�@��?��@���CG��#�@ə�@��A�z�CaH                                    Bxe�I�  
�          AP����RA33@G�A\)C����R@��@�G�A��
C                                      Bxe�Xh  �          AQ��4��@��R@#�
A5��C33�4��@���@��HA�{C n                                    Bxe�g  �          AS33��@�G�@�A�
=C�
��@��@�33B�C�{                                    Bxe�u�  h          AT���
=@�\@�  A���C���
=@���@�G�B��C�{                                    Bxe��Z  �          AX  �4  @��@��HA��RCJ=�4  @@��@�33A�C%�                                    Bxe��   �          AYp��=p�@�33@
�HA(�C� �=p�@�=q@�=qA�p�C�3                                    Bxe���  �          AXz��MG�@xQ�?�33@�{C#0��MG�@/\)@E�AS33C'��                                    Bxe��L  �          AV{�N�H@Vff?h��@z=qC%u��N�H@ ��@=qA%��C)                                    Bxe���  �          AV=q�C�
@��?xQ�@���C�=�C�
@�ff@X��Ak
=C!�                                    Bxe�͘  �          AW
=�2�R@�Q�?��@�z�C�R�2�R@���@�G�A�Q�C�3                                    Bxe��>  �          AT���,��@�{?�=q@��RCk��,��@���@�p�A�  C\)                                    Bxe���  �          AR�H��
@���E��[33C=q��
AQ�?�G�@�Cc�                                    Bxe���  �          AP����HA�\@{A��C�
��H@�33@��AծCu�                                    Bxe�0  �          AO�
�  A�
@��A)p�C�\�  @\@���A�  C�                                    Bxe��  �          AO�
�4��@ƸR?�G�@�p�C33�4��@�
=@���A���CL�                                    Bxe�%|  �          AL(��2ff@��?�ff@��
C� �2ff@�(�@���A�G�Cs3                                    Bxe�4"  �          AF�H�*�\@��ÿJ=q�hQ�C���*�\@�(�@�A-G�C)                                    Bxe�B�  �          AC\)� ��@�Q���z���z�C�{� ��@�G��#33�@Q�C�                                    Bxe�Qn            AEG��&=q@�{�A��iCs3�&=q@�Q�#�
�@  C��                                    Bxe�`            AN=q��\A�\?h��@�p�B��f��\@��@��\A֏\C.                                    Bxe�n�  T          AQG�����A
=@��Aď\B��{����@ÅAz�B<p�C\)                                    Bxe�}`  �          AM�����A(z�@S33Ao�B�����@�z�@��RBz�C                                      Bxe  
�          AK
=���
A  ?�Q�A��C �����
@�@���A�\C��                                    Bxe�  
          ALz���=qA+
=@��HA��B����=q@�z�A�
B/=qB��                                    Bxe©R  �          AI���Q�A.=q@i��A�\)B�ff��Q�@���A33B)��B��{                                    Bxe·�  T          AIp���z�A%G�@�33A��B�8R��z�@�=qA	G�B.  C ��                                    Bxe�ƞ  r          AN{��p�A6{?u@��\B�����p�A  @��HA�\B�3                                    Bxe��D  �          AIG���A)G�?B�\@_\)B�=��A=q@�G�A��
B��
                                    Bxe���  "          AH  ��A)�?�G�@���B���A(�@��A�
=B��                                     Bxe��  !          AG
=��@��=�G�?�\C�
��@ڏ\@y��A�G�C�                                    Bxe�6  
<          AG\)�A�R�L���qp�C���A�R?��
@��RC(�                                    Bxe��  
�          AHz��*{@�\)�Dz��hQ�C���*{@�녾���5CL�                                    Bxe��  T          AH���<��@��
�B�\�aG�C ��<��@l��?�=qA�\C"��                                    Bxe�-(  "          AH  �$��@�
=��ff��{Cٚ�$��@�(�?�33Ap�C5�                                    Bxe�;�  �          AJ{�2�\@��R��ff��C�R�2�\@�?s33@�z�C�                                    Bxe�Jt  J          AF{�߮@�G��{�����C��߮A�>�{?޸RC ��                                    Bxe�Y  �          AE��Q�A����
�ۅCk���Q�A�H�h����G�B�z�                                    Bxe�g�  ,          AD�����H@��(��={C�q���HAp����
��33B�.                                    Bxe�vf  
�          AG33��ff@����p��"��B�W
��ffAff�QG���33B��                                    BxeÅ  
2          AG
=��Q�@�(�� ���X
=C���Q�A�
�أ��(�B���                                    BxeÓ�  �          AF�\���H@�
=�=q�%��CO\���HAG��vff����B��{                                    BxeâX            AUp���(�A,Q��  �=qB�
=��(�A(  @QG�Ag\)B�\)                                    Bxeð�  
�          AU����A'\)�(���7\)B������A&�R@2�\AB{B�(�                                    Bxeÿ�  h          AU���33A,�ÿfff�y��B�.��33A{@�\)A�(�B��                                    Bxe��J  
�          AT������A<(����
���HB�Q�����A.=q@��
A���B��
                                    Bxe���  2          AW�����A@Q쿺�H�˅B�p�����A3\)@��\A�ffB�u�                                    Bxe��  6          AS33��  A3
=������
B����  A<��?�=qA{B��                                    Bxe��<  
�          AUG��У�A3\)��ff����B�\)�У�A*�R@y��A���B��H                                    Bxe��  T          AT�����A��k���  C
=���A�\@���A�
=C�                                    Bxe��  �          AV=q�0��@���?O\)@^�RC�3�0��@�p�@�z�A�
=CǮ                                    Bxe�&.  T          AV�R�9G�@�=q?��R@���Cn�9G�@�(�@�G�A��C)                                    Bxe�4�  "          AW33�;�
@�(�?���@��CxR�;�
@���@�33A�z�Cٚ                                    Bxe�Cz  �          AW33�733@ڏ\?}p�@�
=C0��733@�\)@�ffA�ffCs3                                    Bxe�R   �          AX  �2�\@����(��ǮCp��2�\@�@A Q�CB�                                    Bxe�`�  
�          ATz��ʏ\A1G�>.{?E�B�z��ʏ\AG�@�=qA�
=B��                                    Bxe�ol  h          AT(���z�A-p���
�#
=B��
��z�A)p�@R�\AiB��                                    Bxe�~  �          AO�
�/\)@���(�����C�q�/\)@��
�33�$��C��                                    BxeČ�  "          AO�
�8(�@\)��
=��G�C'Ǯ�8(�@�\)�U��o�
C��                                    Bxeě^  �          AH���6=q?8Q���Q���  C0c��6=q@L����\)���C$J=                                    BxeĪ  h          AJ=q�$Q�@\(����H��  C!}q�$Q�@��
�>{�c�
CW
                                    Bxeĸ�  �          AQ���@�33���
��G�C�q�A�
=#�
>.{C��                                    Bxe��P  �          AR�H�G�@�{�xQ���=qC�R�G�A	��>Ǯ?�Q�C
�\                                    Bxe���  T          AV�R��A
�H�p����z�C	����A�\?z�H@�ffC8R                                    Bxe��  �          AYp���A�R�mp��~�HC�3��A��?��@�33C�)                                    Bxe��B  �          A]���#33A��u��=qC	���#33A�@�
=A�C��                                    Bxe��  T          A]�G�A)@Q�A]G�B����G�@�(�A�B�C��                                    Bxe��  
�          A]���33A,Q�@�{A�33B�u���33@�=qA
=B!{C�                                    Bxe�4  �          A\����A$��@���A�{B�.��@�z�A!p�B<=qC.                                    Bxe�-�  ^          A^�R��ffA-p�@��A�  B�8R��ff@�Ap�B*G�C�                                    Bxe�<�  |          A`���ᙚA;33@Mp�AS\)B�.�ᙚA	A��B  CT{                                    Bxe�K&  �          Ad���ָRAF{@{A\)B���ָRA
=A Q�B�B�p�                                    Bxe�Y�            A_�
�ƸRAB�R@0  A6ffB�\�ƸRA�
A{B�B���                                    Bxe�hr  �          A_����
AP(�?�=q@�  B�����
A*�R@��B{B��
                                    Bxe�w  �          A_33��\)AP�ÿ�Q���B��f��\)A?\)@���A�ffB���                                    BxeŅ�  
�          A^�R���RAN�H����  B�
=���RABff@�G�A�p�B�ff                                    BxeŔd  �          A_33��p�AB�H?   @�B�k���p�A%��@�{AܸRB���                                    Bxeţ
  |          Ab�R����AH��>�  ?��
B�������A,��@���A�33B�W
                                    Bxeű�  �          A`Q����HAF�\�.{�3�B�ff���HAB=q@o\)Axz�B�aH                                    Bxe��V  �          A`����(�AO�
�@  �E��B�L���(�ALz�@r�\Az�RB��)                                    Bxe���  T          A`�����AR=q�z��z�B�Q����AG\)@�p�A�(�B�(�                                    Bxe�ݢ  �          A]���\ANff��  ��Q�B�����\AAp�@��HA�Q�B݊=                                    Bxe��H  
�          A\���У�AA����=q��\)B왚�У�A0��@���A�33B�                                    Bxe���  
�          A]���Q�AUG��)���2ffBĊ=��Q�ANff@�Q�A���B�{                                    Bxe�	�            A\Q���AJff�P  �dz�B��Ϳ��AI��@Z�HAp��B��
                                    Bxe�:  
�          A_��p�A0Q�@�A�B�G��p�A	G�@�RA�ffC5�                                    Bxe�&�  
�          A`z��p�A/�@C�
AJ�\B����p�A z�AG�BG�C8R                                    Bxe�5�  |          Aa���z�AL(�@/\)A4z�B��H��z�A�
A
=B  B��                                    Bxe�D,  �          A_����\AJ�H@(�A!�Bᙚ���\A�A{B�B���                                    Bxe�R�  T          A^�R��\)AEp�@7�A>=qB��
��\)A��A	G�B�B�8R                                    Bxe�ax  
�          A\�����A6{?�@
�HB��)���A=q@�=qA�G�B��)                                    Bxe�p  |          A\(��=qA#�
��p����C�R�=qA{@UAap�C��                                    Bxe�~�  @          A\����\A.ff>�  ?��
B�W
��\A��@�33A��C��                                    Bxeƍj  �          A]p���RA.�R>�ff?�\)B�G���RAQ�@�G�A�ffC@                                     BxeƜ  
          A]�����A2�R?�=q@��
B��\���A��@�A�  C��                                    Bxeƪ�  	�          A]p���A0��@E�AN�\B�\��A��A=qB�C:�                                    Bxeƹ\  �          A_��\)A-p�@
�HA�B���\)A��@�  A��HC^�                                    Bxe��  
n          A`���p�A5p�>\)?z�B��p�Az�@�\)A�=qC�{                                    Bxe�֨            Ad�����HAT(�?�p�@�
=B�{���HA.�R@�(�B=qB�Ǯ                                    Bxe��N  r          Ab�H�z�A,z�n{�qG�C��z�AG�@���A��C�\                                    Bxe���  
�          Aa���%��Ap����  C	���%��A=q@3�
A7�
C
��                                    Bxe��  �          A`z���
Ap�������C�3��
A@@��AF=qCaH                                    Bxe�@  �          A`Q��=qA ���#33�'�
C��=qA (�@0��A6=qC.                                    Bxe��  �          A\  �\)A,������
=B�u��\)A'33@_\)Ak�C +�                                    Bxe�.�  h          A\Q��+�
A  �0  �7\)Cu��+�
Az�?ٙ�@ᙚC�=                                    Bxe�=2  T          A\(��+�
A z��=p��G
=C5��+�
A
=?�33@��
C��                                    Bxe�K�  �          A]� z�A\)�!G��(  C	u�� z�A  @
=A�C	T{                                    Bxe�Z~  
�          Aa���-Ap��1G��7
=Cz��-A	�?�(�@�=qC��                                    Bxe�i$  
�          A_\)��A8  �Tz��^{B��=��A&�\@�
=A�
=B��f                                    Bxe�w�  |          Ac���=qAF�H������{B��f��=qA9G�@�G�A��B�B�                                    Bxeǆp  �          AaG��ÅA>�\��ff��\)B�L��ÅAF�H@  A�B�\)                                    BxeǕ  T          Ad(�� (�Aff?(�@{CO\� (�A��@�A�G�CT{                                    Bxeǣ�  
n          Ad�����A2�H=#�
>#�
C 5����A
=@�=qA�Q�C:�                                    Bxeǲb  |          Af=q�=qA0�ͼ#�
�#�
C�)�=qA@��RA��C�
                                    Bxe��  
2          Ai����A/\)?��@p�CQ����A  @��A�{C(�                                    Bxe�Ϯ  �          Afff�  A(��@��A�C���  @��@��A�  C�                                    Bxe��T  
(          Ak\)���A,(�?�33@�\)C\)���Az�@أ�A��C                                      Bxe���  
�          Alz��
=A2�R>��?z�C��
=A{@��A�{C+�                                    Bxe���  
�          Ak���A,(�?.{@*�HC�
��A(�@�p�A�=qC	�                                    Bxe�
F  �          AmG��-p�A\)?�(�@��C	h��-p�@���@�A͙�C=q                                    Bxe��  T          Al���9G�A�H?��@�\C^��9G�@�=q@��A�{C}q                                    Bxe�'�  T          Am��@��A�\@%�A ��C�H�@��@��@���A�33C:�                                    Bxe�68  T          AmG��G�
@���?��@˅C�G�
@��R@�G�A�\)Cz�                                    Bxe�D�  T          Al���G33@�z��:=q�7
=C(��G33@���?�G�@~{Cs3                                    Bxe�S�  6          Aj�H�7
=A=q�������C(��7
=A=q@N�RAMp�C�                                     Bxe�b*  	�          Al���=��A z��C33�@��C�)�=��A�
?�=q@�ffC^�                                    Bxe�p�  T          Ah(��;
=@�
=���H����CQ��;
=A��<��
=��
C�f                                    Bxe�v            Ab{�733@���?\)�A�CY��733A��?���@��HC�H                                    BxeȎ  �          A`(��6�H@���������C��6�H@�
=���	��C!H                                    BxeȜ�  
�          A]�%�@��������CJ=�%�A�\�(����CQ�                                    Bxeȫh  "          A\(��*=q@޸R���R��(�C�\�*=qA
{��=q���C�R                                    BxeȺ  �          A[��$��@�\)�Ǯ���C�)�$��A��p����C�=                                    Bxe�ȴ  "          A[�
�)�@У������  Cp��)�A(���G���(�CG�                                    Bxe��Z  �          A\(��,  @ָR���H���RC��,  A��������C�                                    Bxe��   �          A[33�)p�@����R���
C�q�)p�A�\��\��\)C�=                                    Bxe���  b          AY�Q�@ᙚ��33����Cu��Q�Aff�\)�Q�C^�                                    Bxe�L  Z          AX����@�{��G���=qC����A�
��Q���p�Cz�                                    Bxe��  
�          A[����@�\)��{��Q�CG����A�\��\��C33                                    Bxe� �  "          Aa��\)AG���ff���RC	�R�\)A)G�����(�C@                                     Bxe�/>  �          Aa���A ���������C	����A*�R�(���C�q                                    Bxe�=�  
x          A\z��(��@�33��33�ɅC�R�(��A
{������C��                                    Bxe�L�  
�          A]���Mp�@Z�H�tz���ffC%)�Mp�@���У��أ�CE                                    Bxe�[0  T          AaG��K
=@����\����C!Ǯ�K
=@��׿�Q���p�C�=                                    Bxe�i�  
�          A^=q�B�\@�Q����R����C���B�\@�
=�޸R��C�q                                    Bxe�x|  |          A\���
=A=q�N{�k�C޸�
=Az�?�(�@�G�C��                                    Bxeɇ"  "          A^�H�陚A%G���p����B�� �陚A:�\>���?�z�B��                                    Bxeɕ�  �          A[
=��p�A ��������Q�B�\��p�A;���=q����B��                                    Bxeɤn  �          AV=q���@�ff��ff�Q�C�����A���Q��c\)C@                                     Bxeɳ  
n          AZ�H�  @��=q�Q�CY��  A���\)���RC�)                                    Bxe���  �          AU����@Mp��	���!ffC!�����@�G���z��ҏ\C�                                     Bxe��`  
�          AV=q��
@�����Q���C�
��
A
=��(���{C
\)                                    Bxe��  �          Af{���
A>=q�Q��UG�B������
A?\)@AG�AD  B�                                    Bxe���  
�          Ag
=��G�AL��?�p�@�B�\)��G�A((�@�z�A���B�                                    Bxe��R             Ak�����A=����z�B�Q�����A5�@�  A�
=B��{                                    Bxe�
�  "          Adz����
A.�R��p����
B�33���
A8��?�\@�{B�G�                                    Bxe��  
�          Ab�\� ��A.ff�g��o33B���� ��A4(�@\)A�RB�.                                    Bxe�(D  
�          Ac33����A:�\�i���p  B�#�����A?
=@%A)�B��                                    Bxe�6�  
�          Ak��hQ�A`��?�=q@ƸRB�  �hQ�A6�HA��B
B�B�                                    Bxe�E�  
�          Aip���33AT�þ�Q쿵B�p���33A<  @�  A��
B��f                                    Bxe�T6  �          Ah(���=qAQ?�G�@�  B��
��=qA(z�A   Bz�B���                                    Bxe�b�  �          Af�H��ffAC�
@�
A�
B�����ffA  A ��B
=B�Q�                                    Bxe�q�  �          Ag
=��
A1�?�p�@�p�C
��
Az�@�
=A�  C�3                                    Bxeʀ(            Af=q��A8zΰ�
��z�B���A*�\@��\A�=qB�Q�                                    Bxeʎ�  
�          Ad����RA$���;��=�Cp���RA&�\@!�A#\)C#�                                    Bxeʝt  d          Ae���У�AC
=?Y��@aG�B�B��У�A"�H@�  A�G�B�8R                                    Bxeʬ  
�          Amp���Q�ARff@0  A+\)B��f��Q�A!p�A{B(�B��                                    